import pandas as pd


def clean_data(df):

    df = df.copy()

    # Remover colunas que não serão utilizadas
    df = df.drop(
        columns=['car_ID', 'symboling']
    )

    # Traduzir nomes das colunas
    colunas_traduzidas = {
        'CarName': 'Nome_do_Carro',
        'fueltype': 'tipo_de_combustível',
        'aspiration': 'aspiração',
        'doornumber': 'número_de_portas',
        'carbody': 'tipo_de_carroceria',
        'drivewheel': 'rodas_motrizes',
        'enginelocation': 'localização_do_motor',
        'wheelbase': 'distância_entre_eixos',
        'carlength': 'comprimento_do_carro',
        'carwidth': 'largura_do_carro',
        'carheight': 'altura_do_carro',
        'curbweight': 'peso_em_ordem_de_marcha',
        'enginetype': 'tipo_de_motor',
        'cylindernumber': 'número_de_cilindros',
        'enginesize': 'tamanho_do_motor',
        'fuelsystem': 'sistema_de_combustível',
        'boreratio': 'diâmetro_do_cilindro',
        'stroke': 'curso_do_pistão',
        'compressionratio': 'taxa_de_compressão',
        'horsepower': 'potência',
        'peakrpm': 'rpm_máximo',
        'citympg': 'consumo_urbano',
        'highwaympg': 'consumo_rodoviário',
        'price': 'preço'
    }

    df = df.rename(
        columns=colunas_traduzidas
    )

    # Separar marca do nome do carro
    df[['Marca', 'Nome_carro']] = (
        df['Nome_do_Carro']
        .str.split(' ', expand=True, n=1)
    )

    df = df.drop(
        columns=['Nome_do_Carro']
    )

    # Corrigir nomes das marcas
    df['Marca'] = df['Marca'].replace({
        'maxda': 'mazda',
        'nissan': 'Nissan',
        'porcshce': 'porsche',
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen'
    })

    return df