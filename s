df = pd.read_sql(sql, connect)
count = 0
try:
    conn = cx_Oracle.connect(user='CC_S', password='1$', dsn = 'msk-as0')
    cursor = conn.cursor()

    for index, row in df.iterrows():
        login = str(row['Логин']).upper()
        tub_num_df = row['Табельный номер']
        
        # Выполнение запроса для проверки существования записи по логину
        cursor.execute("SELECT LOGIN, TUB_NUM FROM STAGE_UNC.EFS_STAFF@cvm WHERE login = :myField", {"myField": login})
        select_result = cursor.fetchall()
        
        if not select_result:
            # Если запись отсутствует, выполняем вставку
            count += 1
            cursor.execute("INSERT INTO STAGE_UNC.EFS_STAFF@cvm(FULL_NAME, LOGIN, TUB_NUM, DATE_INSERT) VALUES (:1, :2, :3, :4)",
                        (str(row['Фамилия Имя Отчество']).upper(), login, tub_num_df, date))
            cursor.execute("INSERT INTO STAGE_UNC.DICT_NPS_OPERATOR@cvm (STAFF_ID, OPERATOR_NAME, DATE_INSERT) VALUES (:1, :2, :3)",
                        (login, str(row['cisco']).upper(), date))
        else:
            # Если запись существует, проверяем табельный номер
            existing_tub_num = select_result[0][1]
            if int(existing_tub_num) != int(tub_num_df):
                # Если табельный номер не соответствует, удаляем записи и добавляем их заново
                cursor.execute("DELETE FROM STAGE_UNC.EFS_STAFF@cvm WHERE login = :myField", {"myField": login})
                cursor.execute("DELETE FROM STAGE_UNC.DICT_NPS_OPERATOR@cvm WHERE STAFF_ID = :myField", {"myField": login})
                
                count += 1
                cursor.execute("INSERT INTO STAGE_UNC.EFS_STAFF@cvm (FULL_NAME, LOGIN, TUB_NUM, DATE_INSERT) VALUES (:1, :2, :3, :4)",
                            (str(row['Фамилия Имя Отчество']).upper(), login, tub_num_df, date))
                cursor.execute("INSERT INTO STAGE_UNC.DICT_NPS_OPERATOR@cvm (STAFF_ID, OPERATOR_NAME, DATE_INSERT) VALUES (:1, :2, :3)",
                            (login, str(row['cisco']).upper(), date))
    cursor.execute("""
        UPDATE STAGE_UNC.EFS_STAFF@cvm
        SET LOGIN = REPLACE(LOGIN, '\', '')
        WHERE INSTR(LOGIN, '\') > 0
    """)
    cursor.execute("""
        UPDATE STAGE_UNC.DICT_NPS_OPERATOR@cvm
        SET STAFF_ID = REPLACE(STAFF_ID, '\', '')
        WHERE INSTR(STAFF_ID, '\') > 0
    """)
    conn.commit()
    conn.close()
    if count >= 1:
        connect_str = lambda: cx_Oracle.connect(user='CC_S', password='1$', dsn = 'msk-as0')
        connect = create_engine("oracle://", creator=connect_str)
        sql = """select * from STAGE_UNC.EFS_STAFF@cvm where date_insert >= trunc(sysdate)"""
        data = pd.read_sql(sql, connect).to_html(index=False)
        mail = outlook.CreateItem(0)
        mail.To = 'AgadullinaAI@ufa.uralsib.ru'
        mail.Cc = 'PopovNi@uralsib.ru'
        mail.Subject = 'Обновление справочников CSI'
        mail.HTMLBody = 'Добрый день!<BR>Обновлены справочники операторов:<BR>STAGE_UNC.EFS_STAFF@cvm<BR>STAGE_UNC.DICT_NPS_OPERATOR@cvm<BR><BR>' + data
        mail.Send()
except Exception as errors:
    mail = outlook.CreateItem(0)
    mail.To = 'TologonovAB@uralsib.ru'
    mail.Subject = 'Ошибка при загрузке CSI'
    mail.HTMLBody = str(errors)
    mail.Send()



