from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status


from api.service.regression import predict_visitors_with_population


@api_view(["GET"])
def getData(request):
    try:
        pop = int(request.query_params.get("pop"))
        if pop < 0:
            raise ValueError("Population should be a positive integer.")
        result = {"visitors": predict_visitors_with_population(pop)}
        return Response(result)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
