from database.postgresDB import init_database

if __name__ == '__main__':
    print('Configuring PostgresSQL...')
    init_database()