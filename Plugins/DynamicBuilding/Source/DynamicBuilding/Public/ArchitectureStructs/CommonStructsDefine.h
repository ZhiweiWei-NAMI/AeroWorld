// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "Materials/MaterialInterface.h"
#include "UObject/ObjectPtr.h"
#include "UObject/ObjectMacros.h"
#include "Math/MathFwd.h"
#include "CommonStructsDefine.generated.h"

class UPCGScatterPointData;

UENUM(BlueprintType)
enum class EDynamicBuildingResidenceElement : uint8
{
	Balcony = 0 UMETA(DisplayName="阳台"),
	Bedroom = 1 UMETA(DisplayName="卧室"),
	Wall = 2 UMETA(DisplayName="墙"),
	Gable = 3 UMETA(DisplayName="山墙"),
	Staircase = 4 UMETA(DisplayName="楼梯间"),
	Roof = 5 UMETA(DisplayName="屋顶")
};

UENUM(BlueprintType)
enum class EDynamicBuildingLanduse : uint8
{
	Commercial UMETA(DisplayName="商业"),
	Construction UMETA(DisplayName="工地"),
	Education UMETA(DisplayName="教育"),
	Fairground UMETA(DisplayName="会展"),
	Industrial UMETA(DisplayName="工业"),
	Residential UMETA(DisplayName="住宅"),
	Retail UMETA(DisplayName="零售"),
	Other UMETA(DisplayName="其他")
};

USTRUCT(BlueprintType)
struct FDynamicBuildingBoxSizeDescription
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "PositionRange", MakeStructureDefaultValue = "(X=0.500000,Y=1.500000)"), Category = "Default")
	FVector2D PositionRange;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "HeightRange", MakeStructureDefaultValue = "(X=3.000000,Y=7.000000)"), Category = "Default")
	FVector2D HeightRange;

	/** 左，下，右，上 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "SideOffsets", MakeStructureDefaultValue = "(X=200.000000,Y=200.000000,Z=200.000000,W=200.000000)"), Category = "Default")
	FVector4 SideOffsets;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Offset", MakeStructureDefaultValue = "(X=-200.000000,Y=300.000000)"), Category = "Default")
	FVector2D Offset;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingFacadeLayout
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Material"), Category = "Default")
	TObjectPtr<UMaterialInterface> Material;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Layout", MakeStructureDefaultValue = "(X=6.000000,Y=4.000000)"), Category = "Default")
	FVector2D Layout;

	FDynamicBuildingFacadeLayout() {}

	FDynamicBuildingFacadeLayout(
		const TObjectPtr<UMaterialInterface> InMaterial,
		const FVector2D& InLayOut
		)
		: Material(InMaterial)
		, Layout(InLayOut) {}
};

USTRUCT(BlueprintType)
struct FDynamicBuildingFacadeStyle
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "DivideLength"), Category = "Default")
	double DivideLength;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "TopLevelDivideLength"), Category = "Default")
	double TopLevelDivideLength;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "BottomLevelDivideLength"), Category = "Default")
	double BottomLevelDivideLength;

	FDynamicBuildingFacadeStyle()
		: DivideLength(300.0)
		, TopLevelDivideLength(300.0)
		, BottomLevelDivideLength(300.0) {}

	FDynamicBuildingFacadeStyle(
		const double InDivideLength,
		const double InTopLevelDivideLength,
		const double InBottomLevelDivideLength
		)
		: DivideLength(InDivideLength)
		, TopLevelDivideLength(InTopLevelDivideLength)
		, BottomLevelDivideLength(InBottomLevelDivideLength) {}

};

USTRUCT(BlueprintType)
struct FDynamicBuildingSinglePanelStyle
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Offsets"), Category = "Default")
	FVector2D Offsets;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Thickness"), Category = "Default")
	double Thickness;

	FDynamicBuildingSinglePanelStyle()
		: Offsets(FVector2D(0.0, 0.0))
		, Thickness(0.0) {}

	FDynamicBuildingSinglePanelStyle(const FVector2D& InOffsets, const double InThickness)
		: Offsets(InOffsets)
		, Thickness(InThickness) {}

};


USTRUCT(BlueprintType)
struct FDynamicBuildingFinGroupStyle
{
	GENERATED_BODY()

public:
	/** 水平节奏，第一个数是主构建起始序号，第二个数是间隔数量 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "HorizontalRhythm", MakeStructureDefaultValue = "(X=2.000000,Y=4.000000)"), Category = "Default")
	FVector2D HorizontalRhythm;

	/** 主立柱 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MainFin"), Category = "Default")
	FDynamicBuildingSinglePanelStyle MainFin;

	/** 次立柱 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "SubFin"), Category = "Default")
	FDynamicBuildingSinglePanelStyle SubFin;

	FDynamicBuildingFinGroupStyle()
		: HorizontalRhythm(2.0, 4.0)
		, MainFin(FVector2D(-20.0, 100.0), 50.0)
		, SubFin(FVector2D(-50.0, 30.0), 10.0) {}
};

USTRUCT(BlueprintType)
struct FDynamicBuildingLouverGroupStyle
{
	GENERATED_BODY()

public:
	/** 垂直节奏，第一个数是主构建起始序号，第二个数是间隔数量 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "VerticalRhythm"), Category = "Default")
	FVector2D VerticalRhythm;

	/** 主遮阳板 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MainLouver"), Category = "Default")
	FDynamicBuildingSinglePanelStyle MainLouver;

	/** 次遮阳板 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "SubLouver"), Category = "Default")
	FDynamicBuildingSinglePanelStyle SubLouver;

	FDynamicBuildingLouverGroupStyle()
		: VerticalRhythm(FVector2D(2.0, 4.0))
		, MainLouver(FVector2D(-20.0, 100.0), 0.0)
		, SubLouver(FVector2D(-20.0, 30.0), 0.0) {}

	FDynamicBuildingLouverGroupStyle(
		const FVector2D& InVerticalRhythm,
		const FDynamicBuildingSinglePanelStyle& InMainLouver,
		const FDynamicBuildingSinglePanelStyle& InSubLouver
		)
		: VerticalRhythm(InVerticalRhythm)
		, MainLouver(InMainLouver)
		, SubLouver(InSubLouver) {}

};

USTRUCT(BlueprintType)
struct FDynamicBuildingParapet
{
	GENERATED_BODY()

public:
	/** 内外偏移 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Offsets", MakeStructureDefaultValue = "(X=-60.000000,Y=0.000000)"), Category = "Default")
	FVector2D Offsets;

	/** 相比Roof向下向上的高度 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "HeightRage", MakeStructureDefaultValue = "(X=0.000000,Y=150.000000)"), Category = "Default")
	FVector2D HeightRage;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingArchiData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "BuildingIndex", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 BuildingIndex;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "OuterContourIndex", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 OuterContourIndex;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "GroundIndex", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 GroundIndex;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Outline", MakeStructureDefaultValue = "None"), Category = "Default")
	TObjectPtr<UPCGScatterPointData> Outline;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Area", MakeStructureDefaultValue = "0.000000"), Category = "Default")
	double Area;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "FloorNum", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 FloorNum;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "LandUse", MakeStructureDefaultValue = "NewEnumerator0"), Category = "Default")
	EDynamicBuildingLanduse LandUse;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingCubeMaterialInfo
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Mat"), Category = "Default")
	TObjectPtr<UMaterialInterface> Mat;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "RotateAngle", MakeStructureDefaultValue = "0.000000"), Category = "Default")
	double RotateAngle;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Size", MakeStructureDefaultValue = "1.000000,1.000000,1.000000"), Category = "Default")
	FVector Size;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingMaterialGroup
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Mats"), Category = "Default")
	TArray<UMaterialInterface*> Mats;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingMainStructStyle_01
{
	GENERATED_BODY()

public:
	/** 转角边宽 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "CornerWidth"), Category = "Default")
	double CornerWidth;

	/** 顶部结构高度，x为底部高度，y为梁高 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "TopStructureHeight"), Category = "Default")
	FVector2D TopStructureHeight;

	/** 内外偏移 */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Offsets"), Category = "Default")
	FVector2D Offsets;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "HasTopStructure"), Category = "Default")
	bool HasTopStructure;

	FDynamicBuildingMainStructStyle_01()
		: CornerWidth(200.0)
		, TopStructureHeight(FVector2D(600.0, 200.0))
		, Offsets(FVector2D(-20.0, 100.0))
		, HasTopStructure(true) {}

	FDynamicBuildingMainStructStyle_01(
		const double InCornerWidth,
		const FVector2D& InTopStructureHeight,
		const FVector2D& InOffsets,
		const bool InHasTopStructure
		)
		: CornerWidth(InCornerWidth)
		, TopStructureHeight(InTopStructureHeight)
		, Offsets(InOffsets)
		, HasTopStructure(InHasTopStructure) {}

};

USTRUCT(BlueprintType)
struct FDynamicBuildingMateriallayout
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "SlotMaterialID", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 SlotMaterialID;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Layout", MakeStructureDefaultValue = "(X=0.000000,Y=0.000000)"), Category = "Default")
	FVector2D Layout;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingMateriaSetup
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MaterialID", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 MaterialID;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Layout", MakeStructureDefaultValue = "(X=0.000000,Y=0.000000)"), Category = "Default")
	FVector2D Layout;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MemberVar_6", MakeStructureDefaultValue = "None"), Category = "Default")
	TObjectPtr<UMaterialInterface> MemberVar_6;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Name", MakeStructureDefaultValue = "None"), Category = "Default")
	FName Name;
};
