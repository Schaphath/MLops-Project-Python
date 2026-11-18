-- Utilisateur PostgreSQL en LECTURE SEULE pour Grafana et le job Evidently.
-- Le monitoring n'a besoin que de lire la table predictions : jamais users (hash des mots de passe).
--
-- À lancer une fois, stack démarrée (remplacer le mot de passe) :
--   docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ro_password="MOT_DE_PASSE"' < Monitoring/sql/readonly_user.sql
-- Puis dans .env : MONITORING_DB_USER=oncoscan_monitoring et MONITORING_DB_PASSWORD=MOT_DE_PASSE

SELECT 'CREATE ROLE oncoscan_monitoring LOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'oncoscan_monitoring')
\gexec

ALTER ROLE oncoscan_monitoring WITH LOGIN PASSWORD :'ro_password';
ALTER ROLE oncoscan_monitoring SET default_transaction_read_only = on;

SELECT format('GRANT CONNECT ON DATABASE %I TO oncoscan_monitoring', current_database())
\gexec
GRANT USAGE ON SCHEMA public TO oncoscan_monitoring;
GRANT SELECT ON predictions TO oncoscan_monitoring;
