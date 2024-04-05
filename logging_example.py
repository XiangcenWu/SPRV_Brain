import logging

FORMAT = '%(asctime)s, %(message)s'
logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    filemode='w',
    format=FORMAT
)

x = 2.343
logging.info(f'This is {x:.2f}')
