from starlette.responses import JSONResponse


class Response:
    @staticmethod
    def ok(message='success', data=None):
        return JSONResponse(status_code=200,
                            content={'code': 200, 'message': message,
                                     'data': data})
    @staticmethod
    def error(message='error', data=None,status_code=500):
        return JSONResponse(status_code=status_code,
                            content={'code': status_code, 'message': message,
                                     'data': data})