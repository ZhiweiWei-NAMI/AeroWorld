// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "DtfGeomStructsDefine.generated.h"

USTRUCT(BlueprintType)
struct FDtfGeomEdge
{
	GENERATED_BODY()

	/*UFUNCTION(BlueprintCallable, Category="DtfGeom")
	FVector2f Vert;

	UFUNCTION(BlueprintCallable, Category="DtfGeom")
	FVector2f Tri;*/

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DtfGeom")
	FVector A;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DtfGeom")
	FVector B;
};

USTRUCT(BlueprintType)
struct FDtfGeomLineSegment
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DtfGeom")
	FVector A;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DtfGeom")
	FVector B;
};

USTRUCT(BlueprintType)
struct FDtfGeomPolyline
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "DtfGeom")
	TArray<FVector> Pt;

	FDtfGeomPolyline() {}

	FDtfGeomPolyline(const TArray<FVector>& InPt)
		: Pt(InPt) {}
};

/** Please add a struct description */
USTRUCT(BlueprintType)
struct FDtfGeomParticleLightInfo
{
	GENERATED_BODY()

public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Location", MakeStructureDefaultValue = "0.000000,0.000000,0.000000"), Category = "DtfGeom")
	FVector Location;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Color", MakeStructureDefaultValue = "(R=0.000000,G=0.000000,B=0.000000,A=0.000000)"), Category = "DtfGeom")
	FLinearColor Color;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Intensity", MakeStructureDefaultValue = "0.000000"), Category = "DtfGeom")
	double Intensity;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Radius", MakeStructureDefaultValue = "0.000000"), Category = "DtfGeom")
	double Radius;
};