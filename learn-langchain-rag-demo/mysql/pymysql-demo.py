import pymysql

arg_kwargs = {
    'host': "xxx",
    'port': 3306,
    'user': 'root',
    'password': "QWERTyuiop123@",
    'database': "test",
    'charset': 'utf8'
}

if __name__ == '__main__':

    # 1.连接数据库，并得到Connection对象
    db = pymysql.connections.Connection(**arg_kwargs)  # pymysql.connections.Connection对象
    # print(db.__dict__)

    # 2.创建数据库的游标
    cur = db.cursor()
    # 3.sql语句
    sql = "select * from user;"
    # 4.执行sql语句（其实是将sql语句提交给mysql数据库执行，执行后返回结果）
    try:
        cur.execute(sql)  # 是一个可迭代对象,返回一个int类型,为Number of affected rows.

        one = cur.fetchone()
        many = cur.fetchmany(2)
        all = cur.fetchall()
        print(one)
        print(many)
        print(all)
    except Exception as e:
        print(e)
        # 查询不需要rollback，因为select不需要commit
    else:
        print("sql执行成功")
    finally:
        cur.close()  # 先关闭cur
        db.close()  # 再关闭db


