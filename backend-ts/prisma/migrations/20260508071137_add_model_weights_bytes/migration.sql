-- AlterTable
ALTER TABLE "model_versions" ADD COLUMN     "weights" BYTEA,
ALTER COLUMN "file_path" SET DEFAULT '';
