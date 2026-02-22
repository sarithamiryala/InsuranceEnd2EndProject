import psycopg2

DATABASE_URL = "postgresql://postgres:ftlVbPfqqNKExM6Z@db.jwkjuzkvckbxffojdefz.supabase.co:5432/postgres?sslmode=require"

try:
    print("Connecting to Supabase PostgreSQL...")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print("Connected ✅")

    # Insert test row
    cur.execute("""
        INSERT INTO claims (
            transaction_id,
            claim_id,
            customer_name,
            policy_number,
            amount,
            claim_type,
            registered_at,
            status,
            claim_registered
        )
        VALUES (
            'test_tx_001',
            'clm_test_001',
            'Test User',
            'pol_test',
            12345,
            'motor',
            '22-02-2026',
            'REGISTERED',
            TRUE
        )
    """)

    conn.commit()

    print("Row inserted successfully 🎉")

    cur.close()
    conn.close()

except Exception as e:
    print("ERROR ❌:", e)