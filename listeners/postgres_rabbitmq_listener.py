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
    def __init__(self):
        # Paramètres de connexion PostgreSQL
        self.pg_host = os.getenv('PG_HOST')
        self.pg_port = os.getenv('PG_PORT')
        self.pg_database = os.getenv('PG_DATABASE')
        self.pg_user = os.getenv('PG_USER')
        self.pg_password = os.getenv('PG_PASSWORD')
        
        # Schéma et table spécifiques
        self.pg_schema = 'ai'  # À ajuster si nécessaire
        self.table_name = 'web_searcher__memory'

        # Paramètres RabbitMQ
        self.rabbitmq_host = os.getenv('RABBITMQ_HOST')
        self.rabbitmq_port = os.getenv('RABBITMQ_PORT')
        self.rabbitmq_user = os.getenv('RABBITMQ_USER')
        self.rabbitmq_password = os.getenv('RABBITMQ_PASSWORD')

        # Queue de notifications
        self.queue_name = 'queue_vinh_test'
        self.notification_channel = 'web_searcher_memory_channel'

    def create_trigger(self, pg_cursor):
        """Créer un trigger pour la table web_searcher__memory"""
        trigger_function = f"""
        CREATE OR REPLACE FUNCTION notify_web_searcher_memory_change()
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

        DROP TRIGGER IF EXISTS web_searcher_memory_insert_trigger ON {self.pg_schema}.{self.table_name};
        CREATE TRIGGER web_searcher_memory_insert_trigger
        AFTER INSERT ON {self.pg_schema}.{self.table_name}
        FOR EACH ROW EXECUTE FUNCTION notify_web_searcher_memory_change();
        """
        pg_cursor.execute(trigger_function)
        logger.info(f"Trigger créé pour la table {self.pg_schema}.{self.table_name}")

    def create_rabbitmq_connection(self):
        """Créer une connexion RabbitMQ avec gestion des erreurs"""
        try:
            connection_params = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=int(self.rabbitmq_port),
                credentials=pika.PlainCredentials(
                    self.rabbitmq_user, 
                    self.rabbitmq_password
                ),
                connection_attempts=3,
                retry_delay=5,
                socket_timeout=5
            )
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            return connection, channel
        except Exception as e:
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

                # Connexion RabbitMQ
                rabbit_connection, rabbit_channel = self.create_rabbitmq_connection()
                if not rabbit_connection or not rabbit_channel:
                    raise Exception("Impossible de se connecter à RabbitMQ")

                # Curseur PostgreSQL
                pg_cursor = pg_conn.cursor()
                
                # Créer le trigger
                self.create_trigger(pg_cursor)
                
                pg_cursor.execute(f"LISTEN {self.notification_channel};")
                
                logger.info(f" Démarrage de l'écoute des notifications PostgreSQL sur le canal {self.notification_channel}...")

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
                                # Publier sur RabbitMQ
                                rabbit_channel.basic_publish(
                                    exchange='',
                                    routing_key=self.queue_name,
                                    body=json.dumps(transformed_payload),
                                    properties=pika.BasicProperties(delivery_mode=2)
                                )
                                
                                logger.info(f" Notification transmise : {transformed_payload}")
                            else:
                                logger.warning(" Payload non transformé, aucune publication")
                        
                        except Exception as e:
                            logger.error(f" Erreur de traitement : {e}")
                    
                    time.sleep(0.1)

            except (psycopg2.Error, pika.exceptions.AMQPError) as e:
                logger.error(f" Erreur de connexion : {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)

# Exécution
if __name__ == "__main__":
    listener = PostgresRabbitMQListener()
    listener.start_listening()
