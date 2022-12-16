# Images Exploration

Este código lê as informações de bbox (*bouding box*) das anotações e as apresenta nas imagens correspondentes para visualização dos dados de detecção.

## Como usar

```bash
python show_images.py -d [DATASET] -n [NUMBER]
```

**DATASET**: Tipo de dataset 'train' ou 'val'; e

**NUMBER**: Número de images para mostrar.

### Exemplo

```bash
python show_images.py -d train -n 10
```