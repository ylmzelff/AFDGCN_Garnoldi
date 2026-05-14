-- CreateTable
CREATE TABLE "kayseri_config" (
    "id" TEXT NOT NULL DEFAULT 'singleton',
    "base_url" VARCHAR(255) NOT NULL,
    "username" VARCHAR(100) NOT NULL,
    "password" VARCHAR(255) NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "kayseri_config_pkey" PRIMARY KEY ("id")
);
