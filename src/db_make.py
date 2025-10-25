import numpy as np
import pandas as pd
from pathlib import Path
import math as mt
PATH = Path(__file__).parent
DATAPATH = PATH.parent.parent / 'dados/'


def first_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['uf'] == 33]
    df = df[df['município'] == 330455]
    df = df[df['cbo2002ocupação'] <= 232170]
    df = df[df['cbo2002ocupação'] >= 232105]
    return df

def second_filter(df: pd.DataFrame) -> pd.DataFrame:
    
    df['salário'] = (
    df['salário']
    .astype(str)
    .str.replace(',', '.', regex=False)
    .replace(['', 'nan', 'None'], None)
    .astype(float)
    )
    
    df = df[df['salário'].fillna(0) != 0]
    
    df['horascontratuais'] = (
    df['horascontratuais']
    .str.replace(',', '.', regex=False)
    .replace('', None)
    .astype(float)
    )
   
    df = df[df['horascontratuais'] != 0]
    df['aula_hora'] = df['salário'] / ((df['horascontratuais'] * 60 / 50)*4)
    
    return df

import pandas as pd
from typing import Dict, List, Any

def third_filter(df: pd.DataFrame) -> pd.DataFrame:
    CBO_PROFESSORS = {
        232110: "Professor de Biologia no Ensino Médio",
        232125: "Professor de Filosofia no Ensino Médio",
        232130: "Professor de Física no Ensino Médio",
        232135: "Professor de Geografia no Ensino Médio",
        232140: "Professor de História no Ensino Médio",
        232145: "Professor de Língua e Literatura Brasileira no Ensino Médio",
        232150: "Professor de Língua Estrangeira Moderna no Ensino Médio",
        232155: "Professor de Matemática no Ensino Médio",
        232160: "Professor de Psicologia no Ensino Médio",
        232165: "Professor de Química no Ensino Médio",
        232170: "Professor de Sociologia no Ensino Médio"
    }

    cbos = list(CBO_PROFESSORS.keys())
    
    df_filtrado = df[df['cbo2002ocupação'].isin(cbos)].copy()
    df_filtrado['aula_hora'] = pd.to_numeric(df_filtrado['aula_hora'], errors='coerce')

    group_basic = df_filtrado.groupby('cbo2002ocupação').agg(
        Q1 = ('aula_hora', lambda x: x.quantile(0.25)),
        Mediana = ('aula_hora', lambda x: x.quantile(0.5)),
        Q3 = ('aula_hora', lambda x: x.quantile(0.75)),
        Média_com = ('aula_hora', 'mean'),
        Desvio_com = ('aula_hora', 'std'),
        Tamanho = ('aula_hora', 'size')
    )

    group_basic['IQR'] = group_basic['Q3'] - group_basic['Q1']
    group_basic['LI'] =(group_basic['Q1'] - 1.5 * group_basic['IQR']).clip(lower=0)
    group_basic['LS'] = group_basic['Q3'] + 1.5 * group_basic['IQR']

    df_filtrado['LI'] = df_filtrado['cbo2002ocupação'].map(group_basic['LI'])
    df_filtrado['LS'] = df_filtrado['cbo2002ocupação'].map(group_basic['LS'])
    
    outlier_mask = (df_filtrado['aula_hora'] < df_filtrado['LI']) | (df_filtrado['aula_hora'] > df_filtrado['LS'])
    df_filtrado['is_outlier'] = outlier_mask.fillna(False)

    outlier_counts = df_filtrado.groupby('cbo2002ocupação')['is_outlier'].sum().rename('Outliers')

    df_no_out = df_filtrado[~df_filtrado['is_outlier']].copy()
    group_no = df_no_out.groupby('cbo2002ocupação').agg(
        Q1_sem = ('aula_hora', lambda x: x.quantile(0.25)),
        Mediana_sem = ('aula_hora', lambda x: x.quantile(0.5)),
        Q3_sem = ('aula_hora', lambda x: x.quantile(0.75)),
        Média_sem = ('aula_hora', 'mean'),
        Desvio_sem = ('aula_hora', 'std'),
        Tamanho_sem = ('aula_hora', 'size')
    )

    estatisticas = group_basic.join(outlier_counts).join(group_no).reset_index()


    all_data_com = df_filtrado['aula_hora']
    
    if all_data_com.empty or all_data_com.isnull().all():
        estatisticas['Título da Ocupação'] = estatisticas['cbo2002ocupação'].map(CBO_PROFESSORS)
        return estatisticas
        
    Q1_geral = all_data_com.quantile(0.25)
    Q3_geral = all_data_com.quantile(0.75)
    IQR_geral = Q3_geral - Q1_geral
    LI_geral = max(0, Q1_geral - 1.5 * IQR_geral)
    LS_geral = Q3_geral + 1.5 * IQR_geral
    
    outlier_mask_geral = (all_data_com < LI_geral) | (all_data_com > LS_geral)
    
   
    all_data_no = all_data_com[~outlier_mask_geral]
    
    linha_geral = {
        'cbo2002ocupação': 0, 
        'Título da Ocupação': 'Geral (Todos os Professores)',
        'Tamanho': all_data_com.size,
        'Outliers': outlier_mask_geral.sum(),
        'Q1': Q1_geral,
        'Mediana': all_data_com.quantile(0.5),
        'Q3': Q3_geral,
        'IQR': IQR_geral,
        'LI': LI_geral,
        'LS': LS_geral,
        'Média_com': all_data_com.mean(),
        'Desvio_com': all_data_com.std(),
        'Tamanho_sem': all_data_no.size,
        'Q1_sem': all_data_no.quantile(0.25),
        'Mediana_sem': all_data_no.quantile(0.5),
        'Q3_sem': all_data_no.quantile(0.75),
        'Média_sem': all_data_no.mean(),
        'Desvio_sem': all_data_no.std(),
    }
    
    df_linha_geral = pd.DataFrame([linha_geral])
    estatisticas = pd.concat([estatisticas, df_linha_geral], ignore_index=True)

    estatisticas['Título da Ocupação'] = estatisticas['cbo2002ocupação'].map(CBO_PROFESSORS).fillna(
        estatisticas['Título da Ocupação']
    )

    cols_order = [
        'cbo2002ocupação', 'Título da Ocupação',
        'Tamanho', 'Outliers', 'Tamanho_sem',
        'Q1', 'Mediana', 'Q3', 'IQR', 'LI', 'LS',
        'Q1_sem', 'Mediana_sem', 'Q3_sem',
        'Média_com', 'Média_sem',
        'Desvio_com', 'Desvio_sem'
    ]
    cols_order = [c for c in cols_order if c in estatisticas.columns]
    estatisticas = estatisticas[cols_order]

    if 'Outliers' in estatisticas.columns and 'Tamanho' in estatisticas.columns:
        estatisticas['% Outliers'] = (estatisticas['Outliers'] / estatisticas['Tamanho']).fillna(0) * 100

    return estatisticas

def _1():
    for idx in range(1, 9):
        df = pd.read_csv(DATAPATH / f'{idx}.csv', sep=';', encoding='utf-8')
        df = first_filter(df)
        df.to_csv(DATAPATH / f'{idx}_filtered.csv', sep=';', encoding='utf-8')

def _2():
    for idx in range(1, 9):
        df = pd.read_csv(DATAPATH / f'{idx}_filtered.csv', sep=';', encoding='utf-8')
        df = second_filter(df)        
        df.to_csv(DATAPATH / f'{idx}_filtered_2.csv', sep=';', encoding='utf-8')

def _3():
    for idx in range(1, 9):
        df = pd.read_csv(DATAPATH / f'{idx}_filtered_2.csv', sep=';', encoding='utf-8')
        df = third_filter(df) 
        df.to_excel(DATAPATH / f'mes_{idx}_analise_final.xlsx', index=False)

def _4():
    df = pd.DataFrame()
    
   
    for idx in range(1, 9): 
        new_df = df = pd.read_csv(DATAPATH / f'{idx}_filtered_2.csv', sep=';', encoding='utf-8')

        df = pd.concat([df, new_df], axis=0)
        
    df = third_filter(df)
    df.to_excel(DATAPATH / f'Captação_de_Recursos.xlsx', index=False)
    return df

def main():
    #_1()
    #_2()
    #_3()
    _4()
    return

if __name__ == '__main__':  
    main()