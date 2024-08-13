#Librerías
import ShoppingIA as shop


def main():
    #Declarar clase
    class_shop = shop.ShopIA()

    #Inicializar clase
    cap = class_shop.init()

    #Stream
    stream = class_shop.tiendaIA(cap)

if __name__ == "__main__":
    main()