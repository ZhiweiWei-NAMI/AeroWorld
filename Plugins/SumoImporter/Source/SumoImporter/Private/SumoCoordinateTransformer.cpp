#include "SumoCoordinateTransformer.h"

FVector FSumoCoordinateTransformer::TransformPointMeters(const FVector& SumoPointMeters, const FSumoTransformConfig& Config)
{
	const FVector AxisMappedMeters = ApplyAxisMapping(SumoPointMeters, Config.AxisMapping);
	const FVector AxisMappedCm = AxisMappedMeters * Config.ScaleCmPerMeter;
	const FRotator YawRotation(0.0f, Config.YawOffsetDeg, 0.0f);
	return YawRotation.RotateVector(AxisMappedCm) + Config.TranslationCm;
}

FVector FSumoCoordinateTransformer::TransformDirectionMeters(const FVector& SumoDirectionMeters, const FSumoTransformConfig& Config)
{
	const FVector AxisMappedDirection = ApplyAxisMapping(SumoDirectionMeters, Config.AxisMapping);
	const FRotator YawRotation(0.0f, Config.YawOffsetDeg, 0.0f);
	return YawRotation.RotateVector(AxisMappedDirection).GetSafeNormal();
}

void FSumoCoordinateTransformer::TransformPointsMeters(
	const TArray<FVector>& InPointsMeters,
	const FSumoTransformConfig& Config,
	TArray<FVector>& OutPointsCm)
{
	OutPointsCm.Reset(InPointsMeters.Num());
	for (const FVector& PointMeters : InPointsMeters)
	{
		OutPointsCm.Add(TransformPointMeters(PointMeters, Config));
	}
}

FVector FSumoCoordinateTransformer::ApplyAxisMapping(const FVector& InPoint, ESumoAxisMapping Mapping)
{
	switch (Mapping)
	{
	case ESumoAxisMapping::XY_To_XY:
		return FVector(InPoint.X, InPoint.Y, InPoint.Z);
	case ESumoAxisMapping::XY_To_XNegY:
		return FVector(InPoint.X, -InPoint.Y, InPoint.Z);
	case ESumoAxisMapping::XY_To_YX:
		return FVector(InPoint.Y, InPoint.X, InPoint.Z);
	case ESumoAxisMapping::XY_To_YNegX:
		return FVector(InPoint.Y, -InPoint.X, InPoint.Z);
	default:
		return InPoint;
	}
}
