import os
import time
import psycopg2
import pika
import json
import logging
import traceback
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresRabbitMQListener:
    def __init__(self, 
                 pg_schema='ai', 
                 table_name='web_searcher__memory', 
                 queue_name='queue_vinh_test', 
                 notification_channel='web_searcher_memory_channel'):
        # Paramètres de connexion PostgreSQL
        self.pg_host = os.getenv('PG_HOST')
        self.pg_port = os.getenv('PG_PORT')
        self.pg_database = os.getenv('PG_DATABASE')
        self.pg_user = os.getenv('PG_USER')
        self.pg_password = os.getenv('PG_PASSWORD')
        
        # Schéma et table spécifiques
        self.pg_schema = pg_schema
        self.table_name = table_name

        # Paramètres RabbitMQ
        self.rabbitmq_host = os.getenv('RABBITMQ_HOST')
        self.rabbitmq_port = os.getenv('RABBITMQ_PORT')
        self.rabbitmq_user = os.getenv('RABBITMQ_USER')
        self.rabbitmq_password = os.getenv('RABBITMQ_PASSWORD')

        # Queue de notifications
        self.queue_name = queue_name
        self.notification_channel = notification_channel

    def create_trigger(self, pg_cursor):
        """Créer un trigger pour la table dynamique"""
        # Noms dynamiques basés sur le nom de la table
        function_name = f"notify_{self.table_name.replace('__', '_')}_change"
        trigger_name = f"{self.table_name}_insert_trigger"

        trigger_function = f"""
        CREATE OR REPLACE FUNCTION {function_name}()
        RETURNS TRIGGER AS $$
        DECLARE
            payload JSON;
        BEGIN
            IF (TG_OP = 'INSERT') THEN
                payload = json_build_object(
                    'operation', 'INSERT',
                    'table', TG_TABLE_NAME,
                    'new_row', row_to_json(NEW)
                );
                PERFORM pg_notify('{self.notification_channel}', payload::text);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS {trigger_name} ON {self.pg_schema}.{self.table_name};
        CREATE TRIGGER {trigger_name}
        AFTER INSERT ON {self.pg_schema}.{self.table_name}
        FOR EACH ROW EXECUTE FUNCTION {function_name}();
        """
        pg_cursor.execute(trigger_function)
        logger.info(f"Trigger {trigger_name} créé pour la table {self.pg_schema}.{self.table_name}")

    def create_rabbitmq_connection(self):
        """Créer une connexion RabbitMQ avec gestion des erreurs avancée"""
        try:
            # Paramètres de connexion
            credentials = pika.PlainCredentials(
                username=self.rabbitmq_user, 
                password=self.rabbitmq_password
            )
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                credentials=credentials,
                connection_attempts=5,  # Nombre de tentatives de connexion
                retry_delay=5,  # Délai entre les tentatives (en secondes)
                socket_timeout=10  # Timeout de socket
            )

            # Création de la connexion
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Déclaration de la queue sans modifier ses paramètres existants
            channel.queue_declare(queue=self.queue_name, durable=True)

            logger.info(f"Connexion RabbitMQ établie sur {self.rabbitmq_host}:{self.rabbitmq_port}")
            return connection, channel

        except (pika.exceptions.AMQPConnectionError, 
                pika.exceptions.AMQPChannelError, 
                ConnectionResetError) as e:
            logger.error(f"Erreur de connexion RabbitMQ : {e}")
            logger.error(traceback.format_exc())
            return None, None

    def transform_payload(self, payload):
        """Transformer le payload en format cible"""
        try:
            # Extraire les informations du payload original
            new_row = payload.get('new_row', {})
            memory = new_row.get('memory', {})
            
            # Construire le nouveau payload
            transformed_payload = {
                'type': 'user',
                'user_id': new_row.get('user_id', ''),
                'user_info': memory.get('memory', '')
            }
            
            return transformed_payload
        except Exception as e:
            logger.error(f"Erreur de transformation du payload : {e}")
            return None

    def start_listening(self):
        while True:
            try:
                # Connexion PostgreSQL
                pg_conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    database=self.pg_database,
                    user=self.pg_user,
                    password=self.pg_password
                )
                pg_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

                # Connexion RabbitMQ avec gestion des erreurs
                rabbit_connection, rabbit_channel = None, None
                max_attempts = 10
                attempt = 0

                while attempt < max_attempts:
                    rabbit_connection, rabbit_channel = self.create_rabbitmq_connection()
                    if rabbit_connection and rabbit_channel:
                        break
                    
                    attempt += 1
                    logger.warning(f"Tentative de connexion RabbitMQ {attempt}/{max_attempts}")
                    time.sleep(10)  # Attente entre les tentatives

                if not rabbit_connection or not rabbit_channel:
                    raise Exception("Impossible de se connecter à RabbitMQ après plusieurs tentatives")

                # Curseur PostgreSQL
                pg_cursor = pg_conn.cursor()
                
                # Créer le trigger
                self.create_trigger(pg_cursor)
                
                pg_cursor.execute(f"LISTEN {self.notification_channel};")
                
                logger.info(f"Démarrage de l'écoute des notifications PostgreSQL sur le canal {self.notification_channel}...")

                while True:
                    pg_conn.poll()
                    
                    while pg_conn.notifies:
                        notify = pg_conn.notifies.pop()
                        
                        try:
                            # Parser le payload JSON
                            payload = json.loads(notify.payload)
                            
                            # Transformer le payload
                            transformed_payload = self.transform_payload(payload)
                            
                            if transformed_payload:
                                # Publier sur RabbitMQ avec gestion des erreurs
                                try:
                                    rabbit_channel.basic_publish(
                                        exchange='',
                                        routing_key=self.queue_name,
                                        body=json.dumps(transformed_payload),
                                        properties=pika.BasicProperties(
                                            delivery_mode=2,  # Message persistant
                                            content_type='application/json'
                                        )
                                    )
                                    
                                    logger.info(f"Notification transmise : {transformed_payload}")
                                
                                except (pika.exceptions.AMQPError, ConnectionResetError) as publish_error:
                                    logger.error(f"Erreur de publication RabbitMQ : {publish_error}")
                                    # Tentative de reconnexion
                                    rabbit_connection, rabbit_channel = self.create_rabbitmq_connection()
                            
                            else:
                                logger.warning("Payload non transformé, aucune publication")
                        
                        except Exception as e:
                            logger.error(f"Erreur de traitement : {e}")
                    
                    time.sleep(0.1)

            except (psycopg2.Error, pika.exceptions.AMQPError, ConnectionResetError) as e:
                logger.error(f"Erreur de connexion : {e}")
                logger.error(traceback.format_exc())
                time.sleep(30)  # Attente plus longue en cas d'erreur

# Exécution
if __name__ == "__main__":
    listener = PostgresRabbitMQListener()
    listener.start_listening()
