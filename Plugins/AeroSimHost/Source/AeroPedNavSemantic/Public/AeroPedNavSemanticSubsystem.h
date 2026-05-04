#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "AeroPedNavSemanticSubsystem.generated.h"

class FJsonObject;

struct FAeroPedAnchorRuntime
{
	FString AnchorId;
	FVector PositionEnuM = FVector::ZeroVector;
	FVector SurfaceNormalEnu = FVector::UpVector;
	TArray<FString> WaitingZoneIds;
};

struct FAeroPedEdgeRuntime
{
	FString EdgeId;
	FString EdgeType;
	FString FromAnchorId;
	FString ToAnchorId;
	TArray<FVector> PolylineEnuM;
	TArray<FVector> PolylineNormalsEnu;
	double LengthM = 0.0;
};

UCLASS()
class AEROPEDNAVSEMANTIC_API UAeroPedNavSemanticSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;

	void SetMapContext(const FString& MapId, const TSharedPtr<FJsonObject>& MapContext);

	bool LoadSemanticSource(const FString& SourcePath, FString& OutError);
	bool LoadSemanticBundle(const FString& BundlePath, FString& OutError);
	bool CompileSemanticBundle(const FString& SourcePath, const FString& BundlePath, FString& OutError);

	TSharedPtr<FJsonObject> QueryPedPath(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const;
	TSharedPtr<FJsonObject> ProjectGround(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const;
	TSharedPtr<FJsonObject> QueryPedAnchor(const TSharedPtr<FJsonObject>& Payload, FString& OutError) const;
	bool ProjectWorldPointToGround(const FVector& InputWorldCm, FVector& OutProjectedWorldCm, FString& OutAnchorId) const;
	bool ProjectWorldPointToGroundDetailed(const FVector& InputWorldCm, FVector& OutProjectedWorldCm, FVector& OutSurfaceNormalWorld, FString& OutAnchorId) const;

private:
	bool RebuildRuntimeCacheFromBundle(const TSharedPtr<FJsonObject>& BundleObject, FString& OutError);
	FVector ConvertEnuMetersToWorldCm(const FVector& PositionEnuM) const;
	bool TryReadVectorField(const TSharedPtr<FJsonObject>& Object, const FString& FieldName, FVector& OutVector) const;
	bool ProjectPointToGroundDetailed(const FVector& InputEnuM, FVector& OutProjectedEnuM, FVector& OutSurfaceNormalEnu, FString& OutAnchorId) const;
	bool ProjectPointToGround(const FVector& InputEnuM, FVector& OutProjectedEnuM, FString& OutAnchorId) const;

private:
	FString CurrentMapId;
	FVector CurrentWorldOriginCm = FVector::ZeroVector;
	TSharedPtr<FJsonObject> SourceDocument;
	TSharedPtr<FJsonObject> BundleDocument;
	TMap<FString, FAeroPedAnchorRuntime> AnchorsById;
	TMap<FString, TArray<FAeroPedEdgeRuntime>> OutgoingEdgesByAnchorId;
	TMap<FString, TSharedPtr<FJsonObject>> WaitingZonesById;
	double MaxSnapDistanceM = 6.0;
	double GroundTraceHalfHeightM = 25.0;
};
