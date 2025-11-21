import numpy as np
from icecream import ic
from datetime import datetime
import pprint
import inspect
import os 

# setting some settings
LOG_DOSYASI = "ExploringDebugging.log"
# pretty view numpy elemnts
np.set_printoptions(precision=4, suppress=True, linewidth=100)

# create turkish time stamp
def turkish_timeStamp():
    simdi = datetime.now()
    
    # static assignment
    aylar = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    
    ay_ismi = aylar[simdi.month]
    
    # Format: 21 Kasım 2025 saat 15.30
    return f"({simdi.day} {ay_ismi} {simdi.year} saat {simdi.strftime('%H.%M')})"

def data_analysis(data, title):
    # fundamental infos
    analiz = {
        "Title": title,
        "Veri Tipi": type(data).__name__ # 'ndarray', 'list', 'dict' vb. yazar
    }

    # numpy matrix
    if isinstance(data, np.ndarray):
        analiz.update({
            "Boyut (Shape)": data.shape,
            "Veri Tipi Detay": str(data.dtype),
            "Istatistikler": {
                "Min": np.min(data),
                "Max": np.max(data),
                "Ortalama": np.mean(data),
                "Non-Zero Sayisi": np.count_nonzero(data)
            },
            # if data greater than 9 then print left 3x3 corner
            "Icerik Onizleme": data[:3, :3].tolist() if data.size > 9 else data.tolist()
        })

    # list, tuple, sets
    elif isinstance(data, (list, tuple, set)):
        analiz.update({
            "Eleman Sayisi (Len)": len(data),
            # show just first 5 element
            "Icerik Onizleme": list(data)[:5] + ["..."] if len(data) > 5 else list(data)
        })

    # dict
    elif isinstance(data, dict):
        analiz.update({
            "Anahtar Sayisi (Len)": len(data),
            "Anahtarlar (Keys)": list(data.keys())[:5] + ["..."] if len(data) > 5 else list(data.keys()),
            # show just first 3 element
            "Ornek Veri (Ilk 3)": {k: data[k] for k in list(data)[:3]}
        })

    # string
    elif isinstance(data, str):
        analiz.update({
            "Karakter Sayisi": len(data),
            # if too long truncate to first 100 char
            "Icerik": data[:100] + "..." if len(data) > 100 else data
        })

    # basic things int, floati bool ...
    elif isinstance(data, (int, float, bool, complex)):
        analiz.update({
            "Deger": data
        })
    
    # unknown like class instance
    else:
        analiz.update({
            "String Temsili": str(data),
            # if have show attributes
            "Ozellikler (__dict__)": getattr(data, '__dict__', "Okunamadi")
        })

    return analiz

def get_caller_info():
    current_frame = inspect.currentframe()
    
    caller_frame = current_frame.f_back.f_back

    full_path = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    
    file_name = os.path.basename(full_path)
    
    return f" <{file_name}: {line_number}> "

def write_debug(title:str, data, write:bool = True):
    if(write):
        data_info = data_analysis(data, title +  get_caller_info())
        ic(data_info)

def formatli_yaz(text):
    # Gelen metnin her satırına 4 boşluk (indent) ekle
    index = text.find("Title")
    index = text.find("'", index + len('"Title"')+1)
    endindex = text.find("'", index + 1)

    title = text[index + 1 : endindex]
    text = text[ : index - len('"Title"') - 2] + text[endindex + 2 :]
    
    title = '"' + title + turkish_timeStamp() + '"'
    indent_str = "    "

    kaydirilmis_metin = "\n".join([indent_str + line for line in text.split('\n')])
    
    with open(LOG_DOSYASI, "a", encoding='utf-8') as f:
        f.write(f"{title} | \n{kaydirilmis_metin}\n\n| \n")

def native_format(obj):
    if isinstance(obj, np.ndarray):
        return str(obj)

    return pprint.pformat(obj, indent=4, width=100, sort_dicts=False)

ic.configureOutput(
    prefix='',
    outputFunction=formatli_yaz,
    argToStringFunction=native_format,
    includeContext=False
)