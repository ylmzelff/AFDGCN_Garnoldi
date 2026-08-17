-- CreateTable
CREATE TABLE "junction_phase_configs" (
    "id" TEXT NOT NULL,
    "city" VARCHAR(50) NOT NULL,
    "region" VARCHAR(50) NOT NULL,
    "junction_id" INTEGER NOT NULL,
    "lanes" JSONB NOT NULL,
    "fixed_yellow" INTEGER NOT NULL DEFAULT 3,
    "fixed_protection" INTEGER NOT NULL DEFAULT 6,
    "min_green" INTEGER NOT NULL DEFAULT 10,
    "cycle_min" INTEGER NOT NULL DEFAULT 60,
    "cycle_max" INTEGER NOT NULL DEFAULT 120,
    "thresh_low" DOUBLE PRECISION NOT NULL DEFAULT 10,
    "thresh_high" DOUBLE PRECISION NOT NULL DEFAULT 50,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "junction_phase_configs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "junction_phase_configs_city_region_idx" ON "junction_phase_configs"("city", "region");

-- CreateIndex
CREATE UNIQUE INDEX "junction_phase_configs_region_junction_id_key" ON "junction_phase_configs"("region", "junction_id");
