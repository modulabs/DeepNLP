'pdf file download'
'x = paper, y = label'


import urllib3
import certifi
import io
import os
import re
from collections import Counter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import time

from urllib.request import urlopen
from bs4 import BeautifulSoup


# 0. download directory create

pdfdirectory = "../data/pdf"
txtdirectory = "../data/txt"

paperdirectory = "../data/paper"
labeldirectory = "../data/label"


if not os.path.exists(pdfdirectory):
    os.makedirs(pdfdirectory)

if not os.path.exists(txtdirectory):
    os.makedirs(txtdirectory)


if not os.path.exists(paperdirectory):
    os.makedirs(paperdirectory)


if not os.path.exists(labeldirectory):
    os.makedirs(labeldirectory)



def deleteBlankCarriageReturn(textData):
    sentences = textData.split('\n')

    retData = ''

    for sentence in sentences:
        if len(sentence) > 1:
            retData += sentence
            retData += "\n"

    return retData


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def download(url):
    # 1. paper pdf download
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "modu-NLP"}

    connection_pool = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

    resp = connection_pool.request('GET', url, headers=headers)
    print(resp.status)

    d1 = resp.data
    d2 = b'PDF unavailable for'

    if resp.status == 200:

        # 제공하지 않은 pdf 인 경우
        if (d1.find(d2) > -1):
            resp.release_conn()
            return

        # 2. pdf save
        head, tail = os.path.split(url)

        pdf_filename = pdfdirectory + "/" + tail

        f = open(pdf_filename, 'wb')
        f.write(resp.data)
        f.close()

        statinfo = os.stat(pdf_filename)

        if (statinfo.st_size < 5000):
            time.sleep(1)
            return

        # 3. pdf -> txt
        textData = convert_pdf_to_txt(pdf_filename)

        # 4. delete BlankCarriageReturn
        textData = deleteBlankCarriageReturn(textData)

        # 5. txt save
        txt_filename = pdf_filename.replace("pdf", "txt")
        f = open(txt_filename, 'w')
        f.write(textData)
        f.close()


        # 6. paper, label
        labelpos = textData.lower().find('abstract')
        print("labelpos = ", labelpos)

        if labelpos >= 0:
            paperpos = textData.lower().find('introduction')
            print("paperpos = ", paperpos)

            # 6. label save
            label_filename = txt_filename.replace("txt", "label")
            f = open(label_filename, 'w')
            labeltext = textData[labelpos + len('abstract'):paperpos]
            # print(labeltext)
            f.write(labeltext)
            f.close()

            # 7. paper save
            paper_filename = label_filename.replace("label", "paper")
            f = open(paper_filename, 'w')
            papertext = textData[paperpos + + len('introduction'):]
            # print(papertext)
            f.write(papertext)
            f.close()

    resp.release_conn()


urlbase = 'https://arxiv.org'
category = "1510"
url = 'https://arxiv.org/list/cs.LG/pastweek?skip=0&show=1000'

html = urlopen(url)
source = html.read()
html.close()

soup = BeautifulSoup(source, "html5lib")

strPDF = '/pdf/'
for a in soup.find_all('a', href=True):
    if a['href'].find(strPDF) == 0:
        print("Found the URL:", a['href'])
        url = urlbase + a['href']
        print(url)
        download(url)

