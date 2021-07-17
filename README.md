# mask-detector
Il software sviluppato si offre come un garante automatizzato del corretto utilizzo della mascherina, 
in grado di stabilire non solo se i soggetti inquadrati indossino o meno la mascherina, ma anche se lo facciano nel modo corretto, 
ampliando così le capacità dei sistemi già esistenti e deresponsabilizzando i proprietari di locali e attività dal controllo agli ingressi, 
o come un semplice reminder da installare di fronte alla propria porta di casa per ricordarsi di uscire protetti.

Per il raggiungimento degli obiettivi sono stati eseguiti i seguenti passi:

- Ricerca, aggregazione ed adeguamento di dati provenienti da fonti diverse
- Addestramento di una Convolutional Neural Network
- Realizzazione di un'applicazione per l'utilizzo del modello

Attenzione: poichè il dataset utilizzato è stato realizzato applicando a ciascun volto uno sticker raffigurante una mascherina, attualmente il sistema riconosce esclusivamente mascherine chirurgiche di colore chiaro.

## Tecnologie utilizzate
- Python 3
- TensorFlow 2.5.0
- OpenCV 4.5.2
- Jupyter

## Esecuzione Notebook (solo in caso di necessità di riprocessamento dei dati o riaddestramento del modello)
1. Eseguire lo script ./install.sh per creare il virtual environment ed installare le librerie necessarie
2. Attivare il virtual environment con il seguente comando: . venv/bin/activate
3. Creare il kernel per Jupyter eseguendo lo script ./create_kernel.sh
4. Avviare Jupyter per eseguire il notebook per il processamento dei dati (prepare_dataset.ipynb) o per l'addestramento del modello (train_CNN.ipynb)

## Esecuzione applicazione
1. Eseguire lo script ./install.sh per creare il virtual environment ed installare le librerie necessarie
2. Attivare il virtual environment con il seguente comando: . venv/bin/activate
3. eseguire main.py con il seguente comando: python3 main.py
4. Per uscire dall'applicazione premere il tasto 'q'
