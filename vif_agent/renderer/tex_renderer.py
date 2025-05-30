import uuid
import PIL.Image
import os
import subprocess
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from loguru import logger



class TexRenderer:

    def __init__(self, debug=False):
        self.debug = debug
        self.cache_path = os.path.join(os.environ.get("HOME"), ".cache/varbench")
        self.rendering_timeout = 30
        pass

    def from_to_file(self, input: str, output: str):
        output_cmd = subprocess.run(
            ["pdflatex", "-halt-on-error", "-output-directory", self.cache_path, input],
            capture_output=True,
        )
        if output_cmd.returncode != 0:
            raise TexRendererException(output_cmd.stderr)

        output_file_name = os.path.join(
            self.cache_path, os.path.basename(input).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        image = convert_from_path(pdf_path=output_file_name)[0]
        image.save(output)

    """ def retry_error_callback(retry_state):
        raise TexRendererException(f"! ==> Fatal error occurred. Renderer stuck, file: {retry_state.args}")

    @retry(stop=stop_after_delay(120), retry_error_callback=retry_error_callback)   """

    def from_string_to_image(self, input_string: str) -> PIL.Image.Image:
        current_temp_file = str(uuid.uuid4()) + ".tex"
        tmp_file_path = os.path.join(self.cache_path, current_temp_file)
        file = open(tmp_file_path, "w")
        file.write(input_string)
        logger.debug("latex renderer writing to " + tmp_file_path)
        file.flush()
        file.close()
        timeout = False
        try:
            output = subprocess.run(
                [
                    "pdflatex",
                    "-halt-on-error",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    self.cache_path,
                    tmp_file_path,
                ],
                timeout=self.rendering_timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            timeout = True
        output_file_name = os.path.join(
            self.cache_path, os.path.basename(tmp_file_path).replace("tex", "pdf")
        )
        if timeout or output.returncode != 0:
            if not self.debug:
                for ext in ["pdf", "tex", "aux", "log"]:
                    todel_file = output_file_name.replace("pdf", ext)
                    os.path.exists(todel_file) and os.remove(todel_file)
            if timeout:
                raise TexRendererException("Timeout reached")
            else:
                raise TexRendererException(
                    output.stderr.decode() + "|" + output.stdout.decode()
                )
        logger.debug(f"converting {tmp_file_path} to png")
        try:
            to_return_image = convert_from_path(pdf_path=output_file_name)[0]
        except PDFPageCountError as pe:
            for ext in ["pdf", "tex", "aux", "log"]:
                todel_file = output_file_name.replace("pdf", ext)
                os.path.exists(todel_file) and os.remove(todel_file)
            raise ImageRenderingException(repr(pe))
        for ext in ["pdf", "tex", "aux", "log"]:
            os.remove(output_file_name.replace("pdf", ext))
        return to_return_image


class TexRendererException(Exception):
    def __init__(self, message: str, *args: object) -> None:
        self.message=message
        super().__init__(message, *args)

    def __str__(self) -> str:
        return f"[TexRendererException:{self.message}]"

    def extract_error(self) -> str:
        import re

        error_lines = []
        start_saving = False
        exception_message = self.message.split("\n")

        for line in exception_message:
            if line.startswith("! "):  # start of error message
                start_saving = True

            if start_saving:
                if line.startswith(
                    "!  ==> Fatal error occurred"
                ):  # end of error message
                    start_saving = False
                    continue
                error_lines.append(line.strip())
        return "\n".join(error_lines)


class ImageRenderingException(TexRendererException):
    def extract_error(self) -> str:
        return self.message
