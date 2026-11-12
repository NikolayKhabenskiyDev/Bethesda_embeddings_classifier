import numpy as np
import pandas as pd
import re
import sentence_transformers as st

def clean_dataset(table):
    template = {
        'Conclusion': [],
        'Description': []
    }
    cleaned_table = pd.DataFrame(template)
    ii = 0
    for index, row in table.iterrows():
        ii = ii + 1
        if pd.isna(row['Conclusion']) == 0:
            conc_text = row['Conclusion']
            conc_text = conc_text.replace('\xa0', ' ')
            conc_text = conc_text.replace('Bethesdа', 'Bethesda')
            conc_text = conc_text.replace('Вethesda', 'Bethesda')
            conc_text = conc_text.replace('Betthesda', 'Bethesda')
            conc_text = conc_text.replace('Bethesd', 'Bethesda')
            conc_text = conc_text.replace('Bethesds', 'Bethesda')
            conc_text = conc_text.replace('Bethеsda', 'Bethesda')
            conc_text = conc_text.replace('Betesda', 'Bethesda')
            conc_text = conc_text.replace('Bethessda', 'Bethesda')
            conc_text = conc_text.replace('Bethsda', 'Bethesda')
            conc_text = conc_text.replace('Beethesda', 'Bethesda')
            conc_text = conc_text.replace('Betehsda', 'Bethesda')
            conc_text = conc_text.replace('Betrhesda', 'Bethesda')
            conc_text = conc_text.replace('Bethesada', 'Bethesda')
            conc_text = conc_text.replace('Bthesda', 'Bethesda')
            conc_text = conc_text.replace('Bdethesda', 'Bethesda')
            conc_text = conc_text.replace('bethesda', 'Bethesda')
            conc_text = conc_text.replace('Bethesdaa', 'Bethesda')
            count = conc_text.count('Bethesda')

            if count == 1:
                if pd.isna(row['Description']):
                    descr_text = ''
                    text_blocks = re.split(r'[.();]+', conc_text)
                    for block in text_blocks:
                        count =  block.count('Bethesda')
                        if count == 0:
                            descr_text += block
                        else:
                            conc_text = block
                else:
                    descr_text = row['Description']
                    descr_text = descr_text.replace('\xa0', ' ')

                if len(descr_text) > 100:
                    new_row = {'Description': descr_text, 'Conclusion': conc_text}
                    cleaned_table = pd.concat([cleaned_table, pd.DataFrame([new_row])])
    return cleaned_table

def identify_marks(conc):
    res = list()
    for cell in conc:
        if pd.isna(cell):
            cell_categ = np.nan
        else:
            cell_categ = ""
            if (re.search(r'категория\s*I\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*I\b", cell) or
                    re.search(r"[( ]?I\sдиагностическая", cell)):
                cell_categ = "I"
            if (re.search(r'категория\s*II\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*II\b", cell) or
                    re.search(r"[( ]?II\s*диагностическая", cell)):
                cell_categ = "II"
            if (re.search(r'категория\s*III\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*III\b", cell) or
                    re.search(r"[( ]?III\s*диагностическая", cell)):
                cell_categ = "III"
            if (re.search(r'категория\s*IV\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*IV\b", cell) or
                    re.search(r"[( ]?IV\s*диагностическая", cell)):
                cell_categ = "IV"
            if (re.search(r'категория\s*V\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*V\b", cell) or
                    re.search(r"[( ]?V\s*диагностическая", cell)):
                cell_categ = "V"
            if (re.search(r'категория\s*VI\b', cell) or
                    re.search(r"Bethesda\s*[-—–]?\s*VI\b", cell) or
                    re.search(r"[( ]?VI\s*диагностическая", cell)):
                cell_categ = "VI"
            if cell_categ == "":
                cell_categ = np.nan

        res.append(cell_categ)
    return res

def create_embeddings(read_name, model, prepared_name, embeddings_name):
    print("cleaning data...")
    excel = pd.read_excel(read_name, usecols=[1, 2])
    excel.columns = ['Conclusion', 'Description']
    excel.drop(labels=[0,1], axis=0, inplace=True)
    clean_data = clean_dataset(excel)
    conclusions = clean_data['Conclusion']
    clean_data['Category'] = identify_marks(conclusions)
    clean_data = clean_data.dropna()
    print("success")

    print("create embeddings...")
    embeddings = model.encode(
        clean_data['Description'].tolist(),
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )
    embeddings = pd.DataFrame(embeddings)
    embed_categ = embeddings
    embed_categ['Category'] = clean_data['Category'].values
    clean_data.to_excel(prepared_name, index=False)
    embed_categ.to_excel(embeddings_name, index=False)
    print("success")
    return embed_categ