import logging
import os

import ingest_lsc54_lsc50
import convert_lsc70
import unify_impute
import spatial_normalization
import export_tensors
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("====================================")
    logging.info("  STARTING END-TO-END LSC PIPELINE")
    logging.info("====================================")
    
    # Paso 1: Ingesta LSC50 y LSC54
    logging.info("\n--- PASO 1: Ingesta LSC50 y LSC54 ---")
    lsc50_ingestor = ingest_lsc54_lsc50.LSC50Ingestor(config.LSC50_LANDMARKS_DIR)
    df_50 = lsc50_ingestor.process()
    if not df_50.empty:
        df_50.to_csv(os.path.join(config.OUTPUT_DIR, "lsc50_interim.csv"), index=False)
    
    lsc54_ingestor = ingest_lsc54_lsc50.LSC54Ingestor(config.LSC54_JSON_PATH)
    df_54 = lsc54_ingestor.process()
    if not df_54.empty:
        df_54.to_csv(os.path.join(config.OUTPUT_DIR, "lsc54_interim.csv"), index=False)
    
    # Paso 2: Convertir LSC70 (Imágenes)
    logging.info("\n--- PASO 2: Extracción Keypoints LSC70 ---")
    converter = convert_lsc70.LSC70Converter(config.LSC70_DIR)
    df_70 = converter.process_all() 
    if not df_70.empty:
        df_70.to_csv(os.path.join(config.OUTPUT_DIR, "lsc70_interim.csv"), index=False)
    
    # Paso 3: Unificar e Imputar
    logging.info("\n--- PASO 3: Unificación e Imputación ---")
    df_imputed = unify_impute.unify_and_impute()
    
    # Paso 4: Normalización Espacial
    logging.info("\n--- PASO 4: Normalización Espacial Invariante ---")
    if not df_imputed.empty:
        spatial_normalization.normalize_spatial(df_imputed)
    else:
        logging.error("Omitiendo normalización porque no hay datos de unificación.")
    
    # Paso 5: Generar Tensores NPY
    logging.info("\n--- PASO 5: Exportación de Tensores LSTM/Transformer ---")
    export_tensors.export_tensors(sequence_length=60)
    
    logging.info("====================================")
    logging.info("  LSC PIPELINE COMPLETADO EXITÓSAMENTE")
    logging.info("====================================")

if __name__ == "__main__":
    main()
