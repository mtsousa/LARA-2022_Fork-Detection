# PyTorch Detection

Esta pasta contém arquivos de referência do PyTorch para treinamento e avaliação de modelos para detecção de objeto. Os códigos apresentados são adaptações dos códigos disponíveis na pasta [detection/](https://github.com/pytorch/vision/tree/main/references/detection) do repositório original.

## Transformações

O arquivo [transforms.py](./transforms.py) implementa algumas classes de transformações de imagem do PyTorch considerando o problema de detecção de objeto.

|     Transformação    |                         Descrição                        |
|:--------------------:|:--------------------------------------------------------:|
|        Buffer        |        Retorna a imagem original sem transformação       |
|        Compose       |           Agrupa diversas transformações juntas          |
|   ConvertImageDtype  |            Modifica o tipo dos dados da imagem           |
|     GaussianBlur     |      Borra a imagem por meio de uma função Gaussiana     |
|     GaussianNoise    | Adiciona ruído à imagem por meio de uma função Gaussiana |
|      PILToTensor     |             Transforma a imagem PIL em tensor            |
|     RandomChoice     |       Escolhe a transformação a partir de uma lista      |
|    RandomGrayscale   |           Converte a imagem para tons de cinza           |
| RandomHorizontalFlip |             Espelha a imagem horizontalmente             |
|     RandomIoUCrop    |         Aproxima a imagem em regiões de interesse        |
|   RandomPerspective  |             Modifica a perspectiva da imagem             |
|    RandomRotation    |                    Rotaciona a imagem                    |
|   RandomTranslation  |                    Translada a imagem                    |
|  RandomVerticalFlip  |              Espelha a imagem verticalmente              |
|     RandomZoomOut    |                      Afasta a imagem                     |
|        Resize        |                   Redimensiona a imagem                  |
|       ToTensor       |              Converte a imagem em um tensor              |

### Exemplos de transformações implementadas

|    **Original**    |   **GaussianNoise**   |
|:------------------:|:---------------------:|
|    ![original]     |       ![noised]       |
|  **RandomIoUCrop** | **RandomPerspective** |
|     ![cropped]     |     ![perspective]    |
| **RandomRotation** | **RandomTranslation** |
|     ![rotated]     |     ![translated]     |

[original]: ./img/original.jpg
[noised]: ./img/noised.jpg
[cropped]: ./img/cropped.jpg
[perspective]: ./img/perspective.jpg
[rotated]: ./img/rotated.jpg
[translated]: ./img/translated.jpg