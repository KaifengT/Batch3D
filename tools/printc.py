import traceback

colorDict = {
    'white'     :97,
    'black'     :30,
    'red'       :31,
    'green'     :32,
    'yellow'    :33,
    'blue'      :34,
    'magenta'   :35,
    'cyan'      :36,
    'lgrey'     :37,
    'dgrey'     :90,
    'lred'      :91,
    'lgreen'    :92,
    'lyellow'   :93,
    'lblue'     :94,
    'lmagenta'  :95,
    'lcyan'     :96,
    
}


def printc(*args, color='white', bold=False, end: str | None = "\n",):
    '''
        'white'   
        'black'   
        'red'     
        'green'   
        'yellow'  
        'blue'    
        'magenta' 
        'cyan'    
        'lgrey'   
        'dgrey'   
        'lred'    
        'lgreen'  
        'lyellow' 
        'lblue'   
        'lmagenta'
        'lcyan'   
    '''
    b = '1;'if bold else ''
    print('\033[' + b + f'{colorDict[color]}m', *args, '\033[0m', end=end)
    
def printt(*args, color='white', bold=False, end: str | None = "\n", width=6, title='', per='', unit=''):
    
    b = '1;'if bold else ''
    t = '{'+f'0:>{width}'+per+'}'+unit
    print('['+'\033[' + b + f'{colorDict[color]}m'+ t.format(title) + '\033[0m]', *args, end=end)
    
def printe():
    print('\033[1;31m')
    traceback.print_exc()
    print('\033[0m')
# if __name__ == '__main__':
#     printt(1, 2, color='red', bold=True, title=0.98, width=4, per='.0%', unit='')