# registration_modifications.py
import SimpleITK as sitk
import numpy as np

def affine_registration(fixed_image_np, fixed_affine, moving_image_np, moving_affine):
    """Perform affine registration using SimpleITK."""
    fixed_image = sitk.GetImageFromArray(fixed_image_np)
    fixed_image.SetOrigin(fixed_affine[:3, 3])
    fixed_image.SetDirection(sitk.GetDirectionFromMatrix(fixed_affine[:3, :3]))
    fixed_image.SetSpacing((1.0, 1.0, 1.0))  # Adjust if necessary

    moving_image = sitk.GetImageFromArray(moving_image_np)
    moving_image.SetOrigin(moving_affine[:3, 3])
    moving_image.SetDirection(sitk.GetDirectionFromMatrix(moving_affine[:3, :3]))
    moving_image.SetSpacing((1.0, 1.0, 1.0))  # Adjust if necessary

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initialize transform.
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration.
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample moving image.
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Convert back to NumPy array.
    resampled_np = sitk.GetArrayFromImage(moving_resampled)

    # Update affine: combining fixed affine with the transform matrix
    transform_matrix = np.array(final_transform.GetMatrix()).reshape(fixed_image.GetDimension(), fixed_image.GetDimension())
    new_affine = fixed_affine.copy()
    new_affine[:fixed_image.GetDimension(), :fixed_image.GetDimension()] = transform_matrix @ new_affine[:fixed_image.GetDimension(), :fixed_image.GetDimension()]
    new_affine[:fixed_image.GetDimension(), 3] = final_transform.GetTranslation()

    return resampled_np, new_affine

def non_rigid_registration(fixed_image_np, fixed_affine, moving_image_np, moving_affine):
    """Perform non-rigid (BSpline) registration using SimpleITK."""
    fixed_image = sitk.GetImageFromArray(fixed_image_np)
    fixed_image.SetOrigin(fixed_affine[:3, 3])
    fixed_image.SetDirection(sitk.GetDirectionFromMatrix(fixed_affine[:3, :3]))
    fixed_image.SetSpacing((1.0, 1.0, 1.0))  # Adjust if necessary

    moving_image = sitk.GetImageFromArray(moving_image_np)
    moving_image.SetOrigin(moving_affine[:3, 3])
    moving_image.SetDirection(sitk.GetDirectionFromMatrix(moving_affine[:3, :3]))
    moving_image.SetSpacing((1.0, 1.0, 1.0))  # Adjust if necessary

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsLBFGSB()
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initialize transform.
    transform_domain_mesh_size = [8]*fixed_image.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, transform_domain_mesh_size)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration.
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample moving image.
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Convert back to NumPy array.
    resampled_np = sitk.GetArrayFromImage(moving_resampled)

    # Update affine: for BSpline, affine remains the same, or you can extract the transform parameters
    new_affine = fixed_affine.copy()

    return resampled_np, new_affine
