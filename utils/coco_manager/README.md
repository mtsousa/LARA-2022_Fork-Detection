# COCO Manager

Este código gerencia as imagens dos datasets COCO e é adaptado de [Immersive Limit](https://github.com/immersive-limit/coco-manager). Para mais informações, acesse https://www.immersivelimit.com/tutorials/how-to-filter-the-coco-dataset-by-category.

## Como usar

### Baixe e extraia o arquivo de anotações

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Filtre as imagens pela categoria

```bash
python filter_coco.py -i [INPUT_FILE] -o [OUTPUT_FILE] -c [CATEGORIES]
```

**INPUT_FILE**: Nome do arquivo de entrada;

**OUTPUT_FILE**: Nome do arquivo de saída; e

**CATEGORIES**: Categorias para filtragem.

#### Exemplo

```bash
python filter_coco.py -i annotations/instances_train2017.json -o annotations/instances_train2017_filtered.json -c person dog cat
```

**Observação**: O filtro inclui todas as imagens que contém ao menos uma das categorias listadas.

## Arquivos de saída

- Arquivo .json com os dados (url, categoria, ground truth) das imagens filtradas;
- Arquivo .txt com a URL das imagens filtradas; e
- Arquivo .txt com o ID das imagens filtradas.