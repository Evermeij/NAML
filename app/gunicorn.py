#for discipl
# bind = '0.0.0.0:80'
# workers = 8
# worker_class = 'sync'
# worker_connections = 1000
# timeout = 1000
# keepalive = 4
#
# errorlog = '-'
# loglevel = 'debug'
# accesslog = '-'
# for local
bind = '0.0.0.0:5000' # for local
workers = 4
worker_class = 'sync'
worker_connections = 1000
timeout = 1000
keepalive = 4

errorlog = '-'
loglevel = 'debug'
accesslog = '-'