# <p align="center">Algoritimo de decisao (Execução dos códigos)
### <p align="center">Trabalho usando arvore de decisão usando Pandas e Jupyter Notebook</p>

#### Preparação do Notebook:
```
  !git clone https://github.com/s0md3v/roop.git
  %cd roop
  !pip install -r requirements.txt
```
```
  !wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx
  !mkdir models
  !mv inswapper_128.onnx ./models
```
```
  !pip uninstall onnxruntime onnxruntime-gpu -y
  !pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118
  !pip install onnxruntime-gpu
```
```
  !python run.py --target /content/bryan.mp4  --source /content/tom.jpg -o /content/swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer
```
![Image](https://i.postimg.cc/JhtKk48s/imagem-2024-12-05-134635443.png)
***

#### Imports de Pandas e das Bibliotecas graficas usados:
```
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn import tree
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split

```
![Image](https://i.postimg.cc/4dvQVKYQ/imagem-2024-12-05-141445556.png)
***

####  Definindo parametros do modelo + Tradução:
```
def input_subSpecie(model):
    sub_specie_translation = {
    'Pygmy three-toed sloth': "Preguiça-anã",
    'Maned three-toed sloth': "Preguiça-de-coleira",
    'Pale-throated sloth': "Preguiça-de-bentinho",
    'Brown-throated sloth': "Preguiça-comum",
    'Linnaeus’s two-toed sloth': "Preguiça-real",
    'Hoffman’s two-toed sloth': "Preguiça-de-Hoffmann"
}

    claw_length_cm = float(input("Largura da garra (cm): "))#[1.75 - 12.2 cm]
    size_cm = float(input("Tamanho (cm): "))#[46.9 - 68.8 cm]
    while True:
            specie = int(input("Espécie \n[1]- Três Dedos \n[2]- Dois dedos\n"))
            if specie == 1 or specie == 2:
                break
            else:
                print("Opção inválida. Por favor, escolha 1 ou 2.")
    tail_length_cm = float(input("Tamanho da cauda (cm): ")) #[-2.94 - 8.54 cm]
    weight_kg = float(input("Peso (kg): "))#[0.95 - 10kg]

    df_usuario = pd.DataFrame({
        'claw_length_cm': [claw_length_cm],
        'size_cm': [size_cm],
        'specie': [specie],
        'tail_length_cm': [tail_length_cm],
        'weight_kg': [weight_kg]
    })

    sub_specie_prediction = model.predict(df_usuario)
    sub_specie_name = sub_specie_prediction[0]

      # Traduzir o nome da subespécie
    sub_specie_name_translated = sub_specie_translation.get(sub_specie_name, sub_specie_name)

    print("\nSua Subespécie foi:", sub_specie_name_translated)
```
![Image](https://i.postimg.cc/BQ7cTfnk/imagem-2024-12-05-141600250.png)
![Image](https://i.postimg.cc/8PYMFdyV/imagem-2024-12-05-141723046.png)
***

####  Define a distibuição do grafico de Barras:
```
def sub_species_distribution(df):
    sub_specie_translation = {
    'Pygmy three-toed sloth': "Preguiça-anã",
    'Maned three-toed sloth': "Preguiça-de-coleira",
    'Pale-throated sloth': "Preguiça-de-bentinho",
    'Brown-throated sloth': "Preguiça-comum",
    'Linnaeus’s two-toed sloth': "Preguiça-real",
    'Hoffman’s two-toed sloth': "Preguiça-de-Hoffmann"
}
    df['sub_specie'] = df['sub_specie'].map(sub_specie_translation)

    # Contagem de subespécies para preguiças de dois e três dedos
    dois_dedos = df[df['specie'] == 2]['sub_specie'].value_counts().sort_index()
    tres_dedos = df[df['specie'] == 1]['sub_specie'].value_counts().sort_index()

    # Combinar as contagens em um único DataFrame // reordenar as colunas para que 'Dois Dedos' venha primeiro
    counts = pd.DataFrame({'Dois Dedos': dois_dedos, 'Três Dedos': tres_dedos})[['Dois Dedos', 'Três Dedos']]

    ordered_index = [
        "Preguiça-de-Hoffmann", "Preguiça-real",  # Dois Dedos
        "Preguiça-de-bentinho", "Preguiça-comum", "Preguiça-de-coleira", "Preguiça-anã"  # Três Dedos
    ]
    counts = counts.reindex(ordered_index)

    # Plotar o gráfico de barras lado a lado
    counts.plot(kind='bar', figsize=(8, 5), color = ['#672aff','#ff3535'])

    plt.title('Distribuição das Subespécies nas Preguiças de Dois e Três Dedos')
    plt.xlabel('Subespécie')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tipo de Preguiça')

    plt.tight_layout()
    plt.show()
```
![Image](https://i.postimg.cc/6QcQZt9G/imagem-2024-12-05-142019920.png)
***

####  Define o Grafico de Dispersão de espécies baseado em Peso x Tamanho: 
```
def dispersion_species(df):
    species_names = {1: 'Três dedos', 2: 'Dois dedos'}
    df['specie'] = df['specie'].map(species_names)

    sns.set_palette("deep")

    sns.scatterplot(x='size_cm', y='weight_kg', hue='specie',palette = ['#ebd1cc','#6e6177'], data=df)
    plt.title('Relação entre Tamanho e Peso por Espécie')
    plt.xlabel('Tamanho (cm)')
    plt.ylabel('Peso (kg)')

    plt.legend(title='Espécies')

    plt.show()
```
![Image](https://i.postimg.cc/906X2fBs/imagem-2024-12-05-142102526.png)
***

####  Define o Grafico de Dispersão de sub-espécies baseado em Peso x Tamanho: 
```
def dispersion_subspecies(df):

    sub_specie_translation = {
    'Pygmy three-toed sloth': "Preguiça-anã",
    'Maned three-toed sloth': "Preguiça-de-coleira",
    'Pale-throated sloth': "Preguiça-de-bentinho",
    'Brown-throated sloth': "Preguiça-comum",
    'Linnaeus’s two-toed sloth': "Preguiça-real",
    'Hoffman’s two-toed sloth': "Preguiça-de-Hoffmann"
}

    df['sub_specie'] = df['sub_specie'].map(sub_specie_translation)
    sns.scatterplot(x='size_cm', y='weight_kg', hue='sub_specie', data=df)
    plt.title('Relação entre Tamanho e Peso por Subespécies')
    plt.xlabel('Tamanho (cm)')
    plt.ylabel('Peso (kg)')
    plt.legend(title='Subespécies')

    plt.show()
```
![Image](https://i.postimg.cc/mhW4q5Ls/imagem-2024-12-05-142247107.png)
***

####  Define o Histograma do comprimento de garra: 
```
def claw_lenght(df):

    df['claw_length_cm'].hist(bins=30)
    plt.title('Distribuição do Comprimento da Garra')
    plt.xlabel('Comprimento da Garra (cm)')
    plt.ylabel('Frequência')
    plt.show()
```
![Image](https://i.postimg.cc/HxjT7MZf/imagem-2024-12-05-142156116.png)
***

####  Definição do CSV e dos dados de teste (80% treino e 20% teste):
```
df = pd.read_csv('sloth_data.csv')
specie_mapping = {name: i for i, name in enumerate(df['specie'].unique(), start=1)}

df['specie'] = df['specie'].map(specie_mapping)

df = df.drop(['index','endangered'], axis=1)
X = df.drop(['sub_specie'],axis=1)
y = df['sub_specie']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_prediction = clf.predict(X_test)
```

####  Debug de previsão e resultados:
```
elif user_choice == 5:
      login = str(input("\nDigite o login: "))
      password = str(input("Digite a senha: "))
      if login == "admin" and password == "admin": #senha e login são 'admin'
        print("\n\033[1mLogin realizado com sucesso!\033[0m")

        while True:
          print("\n\n\033[1mO que deseja escolher\033[0m")
          debug_choice = int(input("\n[1] Distribuição de treinamento\n[2] Predição da arvore de decisão\n[0] Voltar ao menu principal\n\n"))

          if debug_choice == 1:
            print("\n\033[3mPorcentagem usada em cada coluna para a predição\033[0m")
            print("\n\nFeature importances:\n")
            for feature, importance in zip(X.columns, clf.feature_importances_):
                print(f"{feature}: {importance}")

          elif debug_choice == 2:
            print("\n\033[3mDados usados para a predição\033[0m")
            print("\nPrediction for Decision Tree: ",y_prediction)
          elif debug_choice == 0:
              break
          else:
              print("Escolha inválida. Por favor, selecione uma opção válida.")
      else:
        print("\na\033[1mLogin ou senha incorretos, tente novamente...\033[0m")
```
![Image](https://i.ibb.co/zVNwBsF/imagem-2024-12-05-144447759.png)
***
