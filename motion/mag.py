import time
import gpiod


class MagExecutor:
    # JP5 pin 13: GPIO1_A7, exposed as gpiochip1 line 7 (1.8 V logic).
    def __init__(self, chip_name="gpiochip1", line_offset=7):
        self.chip = gpiod.chip(chip_name)
        self.gpio = self.chip.get_line(line_offset)

        config = gpiod.line_request()
        config.consumer = "diansai"
        config.request_type = gpiod.line_request.DIRECTION_OUTPUT

        self.gpio.request(config)
        self.gpio.set_value(1)  # 上电默认关闭电磁铁
        self.closed = False

    def set_value(self, value):
        if value not in (0, 1):
            raise ValueError("GPIO value must be 0 or 1")
        self.gpio.set_value(value)

    def get_value(self):
        return self.gpio.get_value()

    def on(self):
        self.set_value(0)

    def off(self):
        self.set_value(1)

    def close(self):
        if not self.closed:
            try:
                self.off()
            finally:
                self.gpio.release()
                self.closed = True


def main():
    magnet = MagExecutor()

    try:
        for _ in range(10):
            magnet.on()
            print("magnet on")
            time.sleep(2)

            magnet.off()
            print("magnet off")
            time.sleep(2)

    except KeyboardInterrupt:
        print("stopped")

    finally:
        magnet.close()
        print("GPIO released")


if __name__ == "__main__":
    main()
