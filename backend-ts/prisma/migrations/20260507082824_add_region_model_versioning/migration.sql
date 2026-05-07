/*
  Warnings:

  - You are about to drop the `kayseri_config` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "kayseri_config";

-- CreateTable
CREATE TABLE "api_configs" (
    "id" TEXT NOT NULL,
    "city" VARCHAR(50) NOT NULL,
    "base_url" VARCHAR(255) NOT NULL,
    "username" VARCHAR(100) NOT NULL,
    "password" VARCHAR(255) NOT NULL,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "api_configs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "region_configs" (
    "id" TEXT NOT NULL,
    "city" VARCHAR(50) NOT NULL,
    "region" VARCHAR(50) NOT NULL,
    "description" VARCHAR(200) NOT NULL DEFAULT '',
    "junction_ids" JSONB NOT NULL,
    "use_model" BOOLEAN NOT NULL DEFAULT false,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "region_configs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "model_versions" (
    "id" TEXT NOT NULL,
    "name" VARCHAR(100) NOT NULL,
    "description" VARCHAR(500) NOT NULL DEFAULT '',
    "city" VARCHAR(50) NOT NULL,
    "region" VARCHAR(50) NOT NULL,
    "file_path" VARCHAR(500) NOT NULL,
    "num_nodes" INTEGER NOT NULL,
    "lag" INTEGER NOT NULL DEFAULT 1,
    "horizon" INTEGER NOT NULL DEFAULT 1,
    "scaler_mean" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "scaler_std" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "is_active" BOOLEAN NOT NULL DEFAULT false,
    "uploaded_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "model_versions_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "api_configs_city_key" ON "api_configs"("city");

-- CreateIndex
CREATE INDEX "region_configs_city_idx" ON "region_configs"("city");

-- CreateIndex
CREATE UNIQUE INDEX "region_configs_city_region_key" ON "region_configs"("city", "region");

-- CreateIndex
CREATE INDEX "model_versions_city_region_idx" ON "model_versions"("city", "region");

-- CreateIndex
CREATE INDEX "model_versions_is_active_idx" ON "model_versions"("is_active");

-- CreateIndex
CREATE UNIQUE INDEX "model_versions_city_region_name_key" ON "model_versions"("city", "region", "name");
