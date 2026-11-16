-- PostgreSQL Schema - OncoScan v2.1.0

-- 1. TABLE DES UTILISATEURS (Praticiens / Médecins)
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(30) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_users_username UNIQUE (username),
    -- Même règle que USERNAME_PATTERN côté appli, en minuscules
    -- (sinon "Dr.Martin" et "dr.martin" peuvent coexister si on insère hors de l'appli)
    CONSTRAINT chk_username_format CHECK (username ~ '^[a-z0-9_.-]{3,30}$')
);

-- 2. TABLE DES PRÉDICTIONS (Historique des analyses cytologiques)
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,

    -- Praticien ; RESTRICT pour ne jamais perdre l'historique
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,

    -- Identifiant pseudonymisé du patient (pas de nom : données de santé)
    patient_id VARCHAR(50) NOT NULL
        CONSTRAINT chk_patient_id_not_blank CHECK (btrim(patient_id) <> ''),

    -- Les 5 variables cellulaires (mêmes bornes que l'API : à modifier ensemble)
    perimeter DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_perimeter CHECK (perimeter > 40.0 AND perimeter <= 250.0),
    concave_points DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_concave_points CHECK (concave_points >= 0.0 AND concave_points <= 0.3),
    texture DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_texture CHECK (texture > 5.0 AND texture <= 50.0),
    smoothness DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_smoothness CHECK (smoothness > 0.01 AND smoothness <= 0.3),
    symmetry DOUBLE PRECISION NOT NULL
        CONSTRAINT chk_symmetry CHECK (symmetry > 0.05 AND symmetry <= 0.7),

    -- Résultat du modèle ('M' = Maligne, 'B' = Bénigne)
    prediction CHAR(1) NOT NULL
        CONSTRAINT chk_prediction_format CHECK (prediction IN ('M', 'B')),

    -- Probabilité de malignité [0.0 - 1.0]
    probability_malignant DOUBLE PRECISION
        CONSTRAINT chk_probability_range CHECK (
            probability_malignant IS NULL OR probability_malignant BETWEEN 0.0 AND 1.0
        ),

    -- Version du modèle qui a produit la prédiction.
    -- Sans ça, impossible de comparer deux versions ou de retrouver d'où vient une erreur.
    model_version VARCHAR(20) NOT NULL DEFAULT 'inconnue',

    -- Validation par le médecin (base pour mesurer la vraie performance en prod)
    doctor_feedback VARCHAR(15) NOT NULL DEFAULT 'UNCONFIRMED'
        CONSTRAINT chk_doctor_feedback CHECK (doctor_feedback IN ('UNCONFIRMED', 'M', 'B')),
    -- Qui a validé et quand : un feedback sans auteur ne vaut pas grand-chose
    feedback_by BIGINT REFERENCES users(id) ON DELETE RESTRICT,
    feedback_at TIMESTAMPTZ,
    CONSTRAINT chk_feedback_complete CHECK (
        (doctor_feedback = 'UNCONFIRMED' AND feedback_by IS NULL AND feedback_at IS NULL)
        OR (doctor_feedback <> 'UNCONFIRMED' AND feedback_by IS NOT NULL AND feedback_at IS NOT NULL)
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 3. INDEX
-- Historique d'un praticien trié par date (requête principale de l'appli)
CREATE INDEX IF NOT EXISTS idx_predictions_user_date
    ON predictions (user_id, created_at DESC);

-- Suivi d'un patient chez un praticien
-- (avant : index sur patient_id seul, alors que tout est filtré par user_id)
CREATE INDEX IF NOT EXISTS idx_predictions_user_patient
    ON predictions (user_id, patient_id);

-- Prédictions en attente de validation, pour le suivi de performance
CREATE INDEX IF NOT EXISTS idx_predictions_feedback_pending
    ON predictions (created_at DESC)
    WHERE doctor_feedback = 'UNCONFIRMED';