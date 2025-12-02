import rclpy
from rclpy.node import Node
from catchers_vision_interfaces.srv import Speak
import pyttsx3


class SpeakerNode(Node):
    def __init__(self):
        super().__init__('speaker_node')

        self.srv = self.create_service(
            Speak,
            'speak_text',
            self.handle_speak_request
        )

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.get_logger().info("Speaker node ready!")

    def handle_speak_request(self, request, response):
        text = request.text
        self.get_logger().info(f"Speaking: {text}")

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Failed to speak: {e}")
            response.success = False

        return response


def main():
    rclpy.init()
    node = SpeakerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
