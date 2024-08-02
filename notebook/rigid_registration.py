import ants

def register_images(fixed_image_path, moving_image_path):
    # Load the fixed and moving images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)

    # Perform rigid registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='Rigid'
    )

    # Get the transformed moving image
    transformed_image = registration['warpedmovout']

    return registration, transformed_image

# The function can be used in the notebook as follows:
# from rigid_registration import register_images
# registration, transformed_image = register_images(fixed_image_path, moving_image_path)
