
import pymssql as ms
conn=ms.connect(server='127.0.0.1',user='bit2',password='1234',database='db')
cursor=conn.cursor()
cursor.execute("SELECT *FROM iris2;")
row=cursor.fetchone()


while row :
    print("첫컬럼 : %s, 둘컬럼 : %s:" %(row[0],row[1]) )
    row = cursor.fetchone()

conn.close()




# 20-07-14_35
# ssms 데이터 가져오기
#
# import pymssql as ms
#
# conn = ms.connect(server='127.0.0.1', user='bit2', password='3411', database='bitdb')
# # server=localhost 도 가능
#
# cursor = conn.cursor()
#
# cursor.execute('SELECT * FROM iris2;')
#
# # 150행 중 1줄을 가져온다
# row = cursor.fetchone()
# # row = cursor.fetchchone()
# # row = cursor.fetchchone()
#
# while row :
#     print('첫 컬럼 : %s, 둘 컬럼 : %s'%(row[0], row[1]))
#     row = cursor.fetchone()
#
# conn.close()    # connect 했으니 close