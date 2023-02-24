# Download Images

Este código faz o download das imagens do COCO dataset a partir de um arquivo com URLs.

## Como usar

```bash
python download_img.py -i [INPUT_FILE] -d [DATASET]
```

**INPUT_FILE**: Nome do arquivo de entrada; e

**DATASET**: Tipo de dataset 'train', 'val' ou 'test'.

### Exemplo

```bash
python download_img.py -i ../coco_manager/annotations/instances_train2017_url.txt -d train
```

## Arquivos de saída

As imagens baixadas são salvas na pasta do dataset correspondente ('train', 'val' ou 'test') dentro da pasta 'data'.