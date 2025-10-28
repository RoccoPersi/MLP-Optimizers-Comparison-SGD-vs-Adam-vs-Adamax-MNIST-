# Confronto tra Algoritmi di Ottimizzazione per MLP su MNIST

Questo progetto confronta le prestazioni di diversi algoritmi di ottimizzazione applicati a una rete neurale Multi-Layer Perceptron (MLP) addestrata sul dataset MNIST. Gli ottimizzatori confrontati sono Stochastic Gradient Descent (SGD), Adam e AdaMax.

# Descrizione

Il progetto implementa una rete MLP da zero in Python utilizzando NumPy e la addestra sul dataset MNIST (caricato tramite mnist_loader.py, basato sul codice di Michael Nielsen). Vengono create e addestrate istanze della rete MLP con tre diversi ottimizzatori:

Stochastic Gradient Descent (SGD): Implementato nella classe Network nel file mlp.py.

Adam: Implementato nella classe AdamOpt nel file mlp_adam_opt.py.

AdaMax: Implementato nella classe AdaMaxOpt nel file adamax.py.

Il notebook progetto_esame.ipynb orchestra l'esperimento:

Carica il dataset MNIST.

Crea istanze della rete MLP (sia con uno che con due strati nascosti) per ciascun ottimizzatore, specificando i relativi iperparametri (learning rate, beta1, beta2, ecc.).

Addestra ciascuna rete per un numero definito di epoche, misurando il tempo impiegato per epoca e la precisione (accuracy) sul set di test dopo ogni epoca.

Calcola e stampa il tempo totale di addestramento e l'accuratezza media finale per ciascun ottimizzatore.

Genera grafici comparativi utilizzando Matplotlib per visualizzare:

Tempo di addestramento per epoca.

Accuratezza sul set di test per epoca.

# File Inclusi

progetto_esame.ipynb: Il notebook Jupyter principale che esegue l'addestramento e la valutazione comparativa.

mlp.py: Implementazione della rete MLP base con ottimizzatore SGD.

mlp_adam_opt.py: Implementazione della rete MLP con ottimizzatore Adam.

adamax.py: Implementazione della rete MLP con ottimizzatore AdaMax.

mnist_loader.py: (Presumibilmente presente) Utility per caricare e preparare il dataset MNIST (basato sul codice di M. Nielsen).

# Dipendenze

Python 3

NumPy

Matplotlib

gzip (per mnist_loader.py)

pickle (per mnist_loader.py)

Utilizzo

Assicurati di avere tutte le dipendenze installate (puoi usare pip install numpy matplotlib).

Scarica il dataset MNIST (mnist.pkl.gz) e assicurati che sia accessibile da mnist_loader.py (solitamente nella stessa directory o in una sottocartella specificata). Il loader potrebbe anche scaricarlo automaticamente se configurato per farlo.

Assicurati che i file .py (mlp.py, mlp_adam_opt.py, adamax.py, mnist_loader.py) siano nella stessa directory del notebook o nel PYTHONPATH.

Apri ed esegui il notebook progetto_esame.ipynb in un ambiente Jupyter (es. Jupyter Notebook, JupyterLab, Google Colab).

Il notebook eseguirà l'addestramento per ciascun ottimizzatore e per entrambe le architetture di rete (uno e due strati nascosti), mostrando i risultati numerici e i grafici comparativi.

Risultati (Osservati dal Notebook)

Dai risultati mostrati nel notebook, si osserva generalmente che:

Adam tende a convergere più rapidamente verso un'accuratezza elevata rispetto a SGD e AdaMax, specialmente nelle prime epoche. Mantiene buone performance complessive.

AdaMax mostra prestazioni spesso intermedie, migliori di SGD ma potenzialmente meno rapide o leggermente inferiori ad Adam con gli iperparametri scelti.

SGD (con il learning rate semplice utilizzato) converge più lentamente e, nel numero di epoche considerato, raggiunge un'accuratezza finale inferiore rispetto agli ottimizzatori adattivi.

I tempi di addestramento per epoca sono relativamente simili tra i tre ottimizzatori, con piccole variazioni.

L'aggiunta di un secondo strato nascosto può migliorare l'accuratezza finale per tutti gli ottimizzatori, mantenendo tendenze simili nella velocità di convergenza relativa.

Nota: I risultati esatti possono variare leggermente ad ogni esecuzione a causa dell'inizializzazione casuale dei pesi e dello shuffling dei dati durante l'addestramento.
