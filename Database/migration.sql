-- Migration 2.0.0 -> 2.1.0, pour une base déjà créée.
-- Les CREATE TABLE IF NOT EXISTS ne modifient pas une table existante.

BEGIN;

-- Usernames : passer en minuscules avant la contrainte (vérifier qu'il n'y a pas de doublons)
UPDATE users SET username = lower(username) WHERE username <> lower(username);
ALTER TABLE users DROP CONSTRAINT IF EXISTS chk_username_length;
ALTER TABLE users ADD CONSTRAINT chk_username_format
    CHECK (username ~ '^[a-z0-9_.-]{3,30}$');

ALTER TABLE predictions
    ADD CONSTRAINT chk_patient_id_not_blank CHECK (btrim(patient_id) <> '');

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(20) NOT NULL DEFAULT 'inconnue',
    ADD COLUMN IF NOT EXISTS feedback_by BIGINT REFERENCES users(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS feedback_at TIMESTAMPTZ;

ALTER TABLE predictions ADD CONSTRAINT chk_feedback_complete CHECK (
    (doctor_feedback = 'UNCONFIRMED' AND feedback_by IS NULL AND feedback_at IS NULL)
    OR (doctor_feedback <> 'UNCONFIRMED' AND feedback_by IS NOT NULL AND feedback_at IS NOT NULL)
);

DROP INDEX IF EXISTS idx_predictions_label;
DROP INDEX IF EXISTS idx_predictions_patient;
CREATE INDEX IF NOT EXISTS idx_predictions_user_patient
    ON predictions (user_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_feedback_pending
    ON predictions (created_at DESC)
    WHERE doctor_feedback = 'UNCONFIRMED';

COMMIT;