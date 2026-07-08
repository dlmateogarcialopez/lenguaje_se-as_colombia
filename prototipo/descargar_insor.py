"""Descarga videos del diccionario INSOR via WordPress REST API.

Cada palabra tiene 3 videos: sena_video (la seña), definicion_video, ejemplo_video.
Aqui solo bajamos sena_video, que es la seña aislada corta (util para el pipeline).

Uso:
  python descargar_insor.py [limite]

Los videos se guardan en LSCPROPIO/tmp/<slug>/<slug>.m4v — carpeta staging,
el watcher IGNORA 'tmp/' hasta que muevas manualmente cada carpeta a LSCPROPIO/.
"""
import json
import os
import ssl
import sys
import time
import unicodedata
import urllib.request

BASE = "https://educativo.insor.gov.co/wp-json/wp/v2"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LSCPROPIO", "tmp")

# Desactivar verificacion SSL (educativo.insor.gov.co tiene cert quisquilloso).
_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def http(url, intentos=4):
    espera = 1.0
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60, context=_ctx) as r:
                return r.read()
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"       reintentando ({i+1}/{intentos-1}) en {espera:.1f}s: {e}")
            time.sleep(espera)
            espera *= 2


def slug(nombre):
    nfkd = unicodedata.normalize("NFKD", nombre)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return "".join(c for c in s if c.isalnum())


def entradas(per_page=100, pagina=1):
    url = f"{BASE}/diccionario?per_page={per_page}&page={pagina}&_fields=id,title,slug,acf"
    return json.loads(http(url))


def url_video(media_id):
    data = json.loads(http(f"{BASE}/media/{media_id}?_fields=source_url,mime_type"))
    if data.get("mime_type", "").startswith("video/"):
        return data.get("source_url")
    return None


def descargar(url, dest, intentos=3):
    espera = 1.0
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=_ctx) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:
            if i == intentos - 1:
                raise
            time.sleep(espera)
            espera *= 2


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs(DEST, exist_ok=True)
    print(f"[+] descargando hasta {limite} señas del diccionario INSOR -> {DEST}")

    procesadas = 0
    pagina = 1
    while procesadas < limite:
        try:
            batch = entradas(per_page=min(10, limite - procesadas), pagina=pagina)
        except Exception as e:
            print(f"[!] error al listar pagina {pagina}: {e} — saltando pagina")
            pagina += 1
            time.sleep(2)
            if pagina > 100:
                break
            continue
        if not batch:
            break
        for entrada in batch:
            if procesadas >= limite:
                break
            nombre = entrada["title"]["rendered"]
            palabra_slug = slug(nombre)
            acf = entrada.get("acf") or {}
            sena_id = acf.get("sena_video")
            if not sena_id:
                print(f"    [-] {nombre}: sin sena_video, omitido")
                continue

            carpeta = os.path.join(DEST, palabra_slug)
            os.makedirs(carpeta, exist_ok=True)
            existentes = [a for a in os.listdir(carpeta) if a.lower().endswith((".mp4", ".m4v"))]
            if existentes:
                print(f"    [·] {nombre}: ya descargado")
                procesadas += 1
                continue

            try:
                url = url_video(sena_id)
                if not url:
                    print(f"    [-] {nombre}: media no es video")
                    continue
                ext = os.path.splitext(url)[1].lower() or ".mp4"
                dest_path = os.path.join(carpeta, palabra_slug + ext)
                descargar(url, dest_path)
                tam = os.path.getsize(dest_path) / 1024
                print(f"    [OK] {nombre} -> {dest_path} ({tam:.1f}KB)")
                procesadas += 1
                time.sleep(0.3)   # ser cortes con el servidor
            except Exception as e:
                print(f"    [!] {nombre}: {e}")
        pagina += 1

    print(f"\n[+] listo: {procesadas} señas descargadas en {DEST}")


if __name__ == "__main__":
    main()
