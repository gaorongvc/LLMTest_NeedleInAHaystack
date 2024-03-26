import os

from grlibs import mdb


def run():
    table = mdb.db.newrank.wechat.article.list
    query = {}
    sort = [(u"publicTime", -1)]
    cursor = table.find(query, sort=sort, limit=100)
    for doc in cursor:
        try:
            title = doc['title']
            print(title)
            content = doc['content'].replace('。', '。\n')
            file_path = os.path.join('WechatPapers', f"{title}.txt")
            with open(file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(content)
        except:
            continue


if __name__ == '__main__':
    run()
