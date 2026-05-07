-- CreateTable
CREATE TABLE "phase_predictions" (
    "id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "region" VARCHAR(50) NOT NULL,
    "city" VARCHAR(50) NOT NULL DEFAULT 'kayseri',
    "time_label" VARCHAR(10) NOT NULL,
    "minute_index" INTEGER NOT NULL,
    "prediction_source" VARCHAR(30) NOT NULL,
    "kayseri_api_status" VARCHAR(20) NOT NULL,
    "junction_count" INTEGER NOT NULL,
    "total_vehicles" INTEGER NOT NULL DEFAULT 0,
    "payload" JSONB,

    CONSTRAINT "phase_predictions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "model_events" (
    "id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "event_type" VARCHAR(30) NOT NULL,
    "model_path" VARCHAR(255) NOT NULL DEFAULT '',
    "num_nodes" INTEGER,
    "lag" INTEGER,
    "details" VARCHAR(500) NOT NULL DEFAULT '',

    CONSTRAINT "model_events_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "username" VARCHAR(64) NOT NULL,
    "hashed_password" TEXT NOT NULL,
    "disabled" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "phase_predictions_created_at_idx" ON "phase_predictions"("created_at");

-- CreateIndex
CREATE INDEX "phase_predictions_region_idx" ON "phase_predictions"("region");

-- CreateIndex
CREATE INDEX "model_events_created_at_idx" ON "model_events"("created_at");

-- CreateIndex
CREATE INDEX "model_events_event_type_idx" ON "model_events"("event_type");

-- CreateIndex
CREATE UNIQUE INDEX "users_username_key" ON "users"("username");
