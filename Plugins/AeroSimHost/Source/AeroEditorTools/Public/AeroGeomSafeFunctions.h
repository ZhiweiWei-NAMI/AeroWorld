#pragma once

#include "CoreMinimal.h"
#include "UDynamicMesh.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "AeroGeomSafeFunctions.generated.h"

class UStaticMesh;
class UGeometryScriptDebug;

UCLASS()
class AEROEDITORTOOLS_API UAeroGeomSafeFunctions : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Copies mesh data from a StaticMesh into a DynamicMesh.
	 * Tries SourceModel LOD first; if unavailable, falls back to RenderData.
	 * Drop-in replacement for CopyMeshFromStaticMesh that avoids the
	 * "SourceModel LOD is empty" error on imported meshes.
	 */
	UFUNCTION(BlueprintCallable, Category = "AeroGeom|Safe",
		meta = (DisplayName = "Safe Copy Mesh From Static Mesh"))
	static UPARAM(DisplayName = "Dynamic Mesh") UDynamicMesh* SafeCopyMeshFromStaticMesh(
		UStaticMesh* FromStaticMeshAsset,
		UDynamicMesh* ToDynamicMesh,
		bool& bSuccess,
		UGeometryScriptDebug* Debug = nullptr);

	/**
	 * Component-wise vector division with zero-guard.
	 * Returns DefaultValue for any component where the divisor is near-zero.
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "AeroGeom|Safe",
		meta = (DisplayName = "Safe Divide Vector"))
	static FVector SafeDivideVector(
		FVector A,
		FVector B,
		FVector DefaultValue);
};
