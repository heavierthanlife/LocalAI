#!/bin/bash
# Daily PostgreSQL backup - keep last 7 days
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$(dirname "$0")/../backups"
mkdir -p "$BACKUP_DIR"
docker exec localai-postgres pg_dump -U localai localai > "$BACKUP_DIR/$TIMESTAMP.sql"
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
echo "Backup saved: $BACKUP_DIR/$TIMESTAMP.sql"
