import psycopg2

def test_connection():
    try:
        # Paramètres de connexion
        conn = psycopg2.connect(
            dbname="myboun",
            user="p4t",
            password="o3CCgX7StraZqvRH5GqrOFLuzt5R6C",
            host="vps-af24e24d.vps.ovh.net",
            port="30030"
        )
        
        # Créer un curseur
        cur = conn.cursor()
        
        # Tester la connexion
        cur.execute("SELECT 1")
        print("✅ Connexion réussie à la base de données!")
        
        # Tester l'accès à la table web_searcher_memory
        cur.execute("SELECT COUNT(*) FROM web_searcher_memory")
        count = cur.fetchone()[0]
        print(f"✅ Table web_searcher_memory accessible - {count} enregistrements")
        
        # Fermer la connexion
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur de connexion : {str(e)}")

if __name__ == "__main__":
    test_connection()
