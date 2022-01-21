
from .app import Application

if __name__ == '__main__':
    
    a = Application(None)
    a.parse_args()
    a.run()
    
