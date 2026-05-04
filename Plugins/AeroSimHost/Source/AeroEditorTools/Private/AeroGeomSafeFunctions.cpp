#include "AeroGeomSafeFunctions.h"

#include "Engine/StaticMesh.h"
#include "GeometryScript/MeshAssetFunctions.h"
#include "UDynamicMesh.h"

UDynamicMesh* UAeroGeomSafeFunctions::SafeCopyMeshFromStaticMesh(
	UStaticMesh* FromStaticMeshAsset,
	UDynamicMesh* ToDynamicMesh,
	bool& bSuccess,
	UGeometryScriptDebug* Debug)
{
	bSuccess = false;
	if (ToDynamicMesh == nullptr || FromStaticMeshAsset == nullptr)
	{
		return ToDynamicMesh;
	}

	FGeometryScriptCopyMeshFromAssetOptions AssetOptions;
	AssetOptions.bApplyBuildSettings = true;
	AssetOptions.bRequestTangents = true;

	// Try SourceModel LOD first (highest quality, has full edit data).
	FGeometryScriptMeshReadLOD RequestedLOD;
	RequestedLOD.LODType = EGeometryScriptLODType::SourceModel;
	RequestedLOD.LODIndex = 0;

	EGeometryScriptOutcomePins Outcome = EGeometryScriptOutcomePins::Failure;
	UDynamicMesh* Result = UGeometryScriptLibrary_StaticMeshFunctions::CopyMeshFromStaticMesh(
		FromStaticMeshAsset, ToDynamicMesh, AssetOptions, RequestedLOD, Outcome, Debug);

	if (Outcome == EGeometryScriptOutcomePins::Success)
	{
		bSuccess = true;
		return Result;
	}

	// Fallback: use RenderData LOD (always available for imported meshes).
	RequestedLOD.LODType = EGeometryScriptLODType::RenderData;
	RequestedLOD.LODIndex = 0;

	Result = UGeometryScriptLibrary_StaticMeshFunctions::CopyMeshFromStaticMesh(
		FromStaticMeshAsset, ToDynamicMesh, AssetOptions, RequestedLOD, Outcome, Debug);

	bSuccess = (Outcome == EGeometryScriptOutcomePins::Success);
	return Result;
}

FVector UAeroGeomSafeFunctions::SafeDivideVector(FVector A, FVector B, FVector DefaultValue)
{
	constexpr double SafeEpsilon = 1e-8;
	return FVector(
		FMath::Abs(B.X) > SafeEpsilon ? A.X / B.X : DefaultValue.X,
		FMath::Abs(B.Y) > SafeEpsilon ? A.Y / B.Y : DefaultValue.Y,
		FMath::Abs(B.Z) > SafeEpsilon ? A.Z / B.Z : DefaultValue.Z);
}
