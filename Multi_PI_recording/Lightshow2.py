#Used for testing
import time

from pixel_ring import pixel_ring
from gpiozero import LED

power = LED(5)
power.on()

pixel_ring.set_brightness(10)

if __name__ == '__main__':

    pixel_ring.wakeup()
    pixel_ring.speak()
    time.sleep(4)
    pixel_ring.off()


power.off()
