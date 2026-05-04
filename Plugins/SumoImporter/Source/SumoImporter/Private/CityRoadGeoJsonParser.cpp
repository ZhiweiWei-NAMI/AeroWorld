#include "CityRoadGeoJsonParser.h"

#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SumoImporterLog.h"

namespace
{
constexpr double OriginShift = 20037508.342789244;

bool LoadJsonObject(const FString& FilePath, TSharedPtr<FJsonObject>& OutObject)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		return false;
	}

	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Content);
	return FJsonSerializer::Deserialize(Reader, OutObject) && OutObject.IsValid();
}

void LonLatToWebMercator(double LonDeg, double LatDeg, double& OutX, double& OutY)
{
	const double ClampedLat = FMath::Clamp(LatDeg, -85.05112878, 85.05112878);
	OutX = LonDeg * OriginShift / 180.0;
	const double Rad = FMath::DegreesToRadians(ClampedLat);
	OutY = FMath::Loge(FMath::Tan(PI * 0.25 + 0.5 * Rad)) * OriginShift / PI;
}

bool TryGetBoundsCenterFromFile(const FString& BoundsPath, FVector2D& OutCenterMercatorM)
{
	if (!FPaths::FileExists(BoundsPath))
	{
		return false;
	}

	TSharedPtr<FJsonObject> RootObject;
	if (!LoadJsonObject(BoundsPath, RootObject))
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Features = nullptr;
	if (!RootObject->TryGetArrayField(TEXT("features"), Features) || Features == nullptr || Features->Num() == 0)
	{
		return false;
	}

	const TSharedPtr<FJsonObject> FeatureObject = (*Features)[0]->AsObject();
	if (!FeatureObject.IsValid())
	{
		return false;
	}

	if (!FeatureObject->HasTypedField<EJson::Object>(TEXT("properties")))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> PropertiesObject = FeatureObject->GetObjectField(TEXT("properties"));
	if (!PropertiesObject.IsValid())
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* BBox = nullptr;
	if (!PropertiesObject->TryGetArrayField(TEXT("bbox"), BBox) || BBox == nullptr || BBox->Num() < 4)
	{
		return false;
	}

	const double MinX = (*BBox)[0]->AsNumber();
	const double MinY = (*BBox)[1]->AsNumber();
	const double MaxX = (*BBox)[2]->AsNumber();
	const double MaxY = (*BBox)[3]->AsNumber();
	OutCenterMercatorM = FVector2D(0.5 * (MinX + MaxX), 0.5 * (MinY + MaxY));
	return true;
}

bool TryGetBoundsCenter(
	const FString& RoadGeoJsonPath,
	const FString& OptionalBoundsGeoJsonPath,
	FVector2D& OutCenterMercatorM)
{
	if (!OptionalBoundsGeoJsonPath.IsEmpty())
	{
		if (TryGetBoundsCenterFromFile(OptionalBoundsGeoJsonPath, OutCenterMercatorM))
		{
			return true;
		}

		UE_LOG(
			LogSumoImporter,
			Warning,
			TEXT("CityRoadGeoJsonParser: explicit bounds file is invalid or missing bbox, fallback to sibling bounds.geojson: %s"),
			*OptionalBoundsGeoJsonPath);
	}

	const FString RoadDir = FPaths::GetPath(RoadGeoJsonPath);
	const FString CacheDir = FPaths::GetPath(RoadDir);
	const FString BoundsPath = FPaths::Combine(CacheDir, TEXT("bounds.geojson"));
	return TryGetBoundsCenterFromFile(BoundsPath, OutCenterMercatorM);
}

bool TryGetLineStringCoordinates(const TSharedPtr<FJsonObject>& FeatureObject, const TArray<TSharedPtr<FJsonValue>>*& OutCoordinates)
{
	OutCoordinates = nullptr;
	if (!FeatureObject.IsValid())
	{
		return false;
	}

	if (!FeatureObject->HasTypedField<EJson::Object>(TEXT("geometry")))
	{
		return false;
	}

	const TSharedPtr<FJsonObject> GeometryObject = FeatureObject->GetObjectField(TEXT("geometry"));
	if (!GeometryObject.IsValid())
	{
		return false;
	}

	FString GeometryType;
	if (!GeometryObject->TryGetStringField(TEXT("type"), GeometryType))
	{
		return false;
	}

	if (!GeometryType.Equals(TEXT("LineString"), ESearchCase::IgnoreCase))
	{
		return false;
	}

	return GeometryObject->TryGetArrayField(TEXT("coordinates"), OutCoordinates);
}
}

bool FCityRoadGeoJsonParser::ParseFile(
	const FString& FilePath,
	const FString& BoundsGeoJsonPath,
	FSumoNetData& OutNetData,
	FSumoImportStats& OutStats,
	FString& OutError)
{
	OutNetData = FSumoNetData();
	OutStats = FSumoImportStats();
	OutError.Reset();

	TSharedPtr<FJsonObject> RootObject;
	if (!LoadJsonObject(FilePath, RootObject))
	{
		OutError = FString::Printf(TEXT("Invalid GeoJSON file: %s"), *FilePath);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Features = nullptr;
	if (!RootObject->TryGetArrayField(TEXT("features"), Features) || Features == nullptr)
	{
		OutError = TEXT("GeoJSON missing 'features' array.");
		return false;
	}

	FVector2D MercatorCenterM = FVector2D::ZeroVector;
	bool bHasCenter = TryGetBoundsCenter(FilePath, BoundsGeoJsonPath, MercatorCenterM);

	double MinX = TNumericLimits<double>::Max();
	double MinY = TNumericLimits<double>::Max();
	double MaxX = TNumericLimits<double>::Lowest();
	double MaxY = TNumericLimits<double>::Lowest();
	double MinZ = TNumericLimits<double>::Max();
	double MaxZ = TNumericLimits<double>::Lowest();
	bool bHasAnyZ = false;

	TArray<TPair<int32, TArray<FVector>>> DeferredShapes;
	DeferredShapes.Reserve(Features->Num());

	for (int32 FeatureIndex = 0; FeatureIndex < Features->Num(); ++FeatureIndex)
	{
		const TSharedPtr<FJsonObject> FeatureObject = (*Features)[FeatureIndex]->AsObject();
		const TArray<TSharedPtr<FJsonValue>>* Coordinates = nullptr;
		if (!TryGetLineStringCoordinates(FeatureObject, Coordinates) || Coordinates == nullptr || Coordinates->Num() < 2)
		{
			continue;
		}

		TArray<FVector> MercatorPoints;
		MercatorPoints.Reserve(Coordinates->Num());
		for (const TSharedPtr<FJsonValue>& CoordinateValue : *Coordinates)
		{
			const TArray<TSharedPtr<FJsonValue>>& CoordArray = CoordinateValue->AsArray();
			if (CoordArray.Num() < 2)
			{
				continue;
			}

			const double Lon = CoordArray[0]->AsNumber();
			const double Lat = CoordArray[1]->AsNumber();
			const double ZMeters = CoordArray.Num() >= 3 ? CoordArray[2]->AsNumber() : 0.0;
			double MX = 0.0;
			double MY = 0.0;
			LonLatToWebMercator(Lon, Lat, MX, MY);
			MinX = FMath::Min(MinX, MX);
			MinY = FMath::Min(MinY, MY);
			MaxX = FMath::Max(MaxX, MX);
			MaxY = FMath::Max(MaxY, MY);
			MinZ = FMath::Min(MinZ, ZMeters);
			MaxZ = FMath::Max(MaxZ, ZMeters);
			bHasAnyZ = true;
			MercatorPoints.Add(FVector(MX, MY, ZMeters));
		}

		if (MercatorPoints.Num() >= 2)
		{
			DeferredShapes.Emplace(FeatureIndex, MoveTemp(MercatorPoints));
		}
	}

	if (!bHasCenter)
	{
		if (!DeferredShapes.IsEmpty())
		{
			MercatorCenterM = FVector2D(0.5 * (MinX + MaxX), 0.5 * (MinY + MaxY));
			bHasCenter = true;
		}
	}

	if (!bHasCenter || DeferredShapes.IsEmpty())
	{
		OutError = TEXT("No valid LineString geometry found in GeoJSON.");
		return false;
	}

	for (const TPair<int32, TArray<FVector>>& ShapeItem : DeferredShapes)
	{
		const int32 FeatureIndex = ShapeItem.Key;
		const TArray<FVector>& MercatorPoints = ShapeItem.Value;

		FSumoEdgeData EdgeData;
		EdgeData.EdgeId = FString::Printf(TEXT("cg_edge_%d"), FeatureIndex);
		EdgeData.Function = TEXT("normal");
		EdgeData.EdgeType = TEXT("city_road");

		FSumoLaneData LaneData;
		LaneData.EdgeId = EdgeData.EdgeId;
		LaneData.LaneId = FString::Printf(TEXT("%s_0"), *EdgeData.EdgeId);
		LaneData.LaneIndex = 0;
		LaneData.SpeedMps = 0.0f;
		LaneData.ShapePointsM.Reserve(MercatorPoints.Num());

		double LengthMeters = 0.0;
		FVector PreviousLocal = FVector::ZeroVector;
		for (int32 PointIndex = 0; PointIndex < MercatorPoints.Num(); ++PointIndex)
		{
			const FVector& MercatorPoint = MercatorPoints[PointIndex];
			const FVector LocalMeters(
				MercatorPoint.X - MercatorCenterM.X,
				MercatorPoint.Y - MercatorCenterM.Y,
				MercatorPoint.Z);

			if (PointIndex > 0)
			{
				LengthMeters += FVector::Dist(PreviousLocal, LocalMeters);
			}
			PreviousLocal = LocalMeters;
			LaneData.ShapePointsM.Add(LocalMeters);
		}

		LaneData.LengthM = static_cast<float>(LengthMeters);
		EdgeData.Lanes.Add(MoveTemp(LaneData));
		OutNetData.Edges.Add(MoveTemp(EdgeData));
	}

	OutStats.TotalEdges = Features->Num();
	OutStats.ImportedEdges = OutNetData.Edges.Num();
	OutStats.ImportedLanes = OutNetData.Edges.Num();
	OutStats.SkippedEdges = OutStats.TotalEdges - OutStats.ImportedEdges;
	OutStats.JunctionCount = 0;
	OutStats.ConnectionCount = 0;

	UE_LOG(
		LogSumoImporter,
		Log,
		TEXT("Parsed CityGenerator road GeoJSON: features=%d importedEdges=%d importedLanes=%d hasZ=%s zRange=[%.3f, %.3f]"),
		OutStats.TotalEdges,
		OutStats.ImportedEdges,
		OutStats.ImportedLanes,
		bHasAnyZ ? TEXT("true") : TEXT("false"),
		bHasAnyZ ? MinZ : 0.0,
		bHasAnyZ ? MaxZ : 0.0);

	return true;
}
