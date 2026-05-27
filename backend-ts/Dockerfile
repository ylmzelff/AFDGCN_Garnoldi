FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends openssl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
COPY prisma ./prisma/

RUN npm ci

RUN npx prisma generate

COPY . .

RUN npm run build

# HuggingFace Spaces port 7860 kullanır, diğer ortamlar PORT env var'ı kullanır
EXPOSE 7860

# Migration'ları çalıştır, sonra sunucuyu başlat
CMD ["sh", "-c", "npx prisma migrate deploy && node dist/server.js"]
