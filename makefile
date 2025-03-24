.PHONY: start_db clean_db stop_db server

start_db:
	sudo mkdir -p /var/lib/postgresql/data/
	sudo chown -R 999:999 /var/lib/postgresql/data/
	sudo docker compose up -d

stop_db:
	sudo docker compose down

clean_db:
	sudo docker compose down
	sudo rm -rf /var/lib/postgresql/data/

server:
	uvicorn server:app --host 0.0.0.0 --port 8000 --reload