# FRT_Recognition
This module is used for image recognition. We have a directory of known faces which are registered from another program.

### Endpoint: 
POST /recognize_face 
### Sample request:
>{
>    "image": "data:image/jpeg;base64,*{{image_to_recognize}}*"
>}
image_to_recognize is the base64 encoded image.
### Sample response:
{
    "faces": [
        {
            "confidence": 0.8019162356853485,
            "name": "A"
        },
        {
            "confidence": 0.1174662858247757,
            "name": null
        },
        {
            "confidence": 0.4006980061531067,
            "name": null
        }
    ]
}
