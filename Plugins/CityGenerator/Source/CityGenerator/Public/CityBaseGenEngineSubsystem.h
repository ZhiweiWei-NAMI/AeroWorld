// Copyright (C) Alibaba.inc. ChengWei, 2024. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "Interfaces/IHttpRequest.h"
#include "alibabacloud/oss/Types.h"
#include "CityBase/CityBaseGenerator.h"
#include "Misc/DateTime.h"
#include "CityBaseGenEngineSubsystem.generated.h"
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnTileDownloadFinished,bool,bSuccess, FString, Code);
DECLARE_DELEGATE_OneParam(FOnGenerateFinished, const FString&);

class UPCGComponent;

USTRUCT(BlueprintType)
struct FCityGenParam
{
	GENERATED_BODY()
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "配置参数", DisplayName = "是否需要烘焙")
	bool bAutoCook = false;
	/// <summary>
	/// 如果为true，则自动计算中心点
	/// 如果为false，则使用Center指定的中心点
	/// </summary>
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "配置参数", DisplayName = "自动计算中心点")
	bool bAutoCenter=true;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "配置参数", DisplayName = "指定中心点")
	FVector2D Center=FVector2D(113.26881,23.13006);
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "道路参数", DisplayName = "道路参数")
	FRoadParams RoadParams;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "道路参数", DisplayName = "道路材质")
	TArray<UMaterialInterface*> RoadMaterialArray;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "地形参数", DisplayName = "地形参数")
	FTerrainParams TerrainParams;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "地形参数", DisplayName = "地形材质")
	TArray<UMaterialInterface*> TerrainMaterialArray;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "地面参数", DisplayName = "地面参数")
	TArray<UMaterialInterface*> GroundMaterialArray;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "水面参数", DisplayName = "水面参数")
	TArray<UMaterialInterface*> WaterMaterialArray;

};
/**
 *
 */
UCLASS()
class CITYGENERATOR_API UCityBaseGenEngineSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()
public:

	UFUNCTION(BlueprintCallable, Category = "CityBaseGen")
	bool ImportFiles(const FString& Content);

	UFUNCTION(BlueprintCallable, Category = "CityBaseGen")
	bool ImportCode(const FString& Code);

	UFUNCTION(BlueprintCallable, Category = "CityBaseGen")
	void PCGCleanupImmediate(UPCGComponent* PCGComponent, bool bRemoveComponents);

	/// <summary>
	/// js call this function to download tile
	/// </summary>
	/// <param name="Content"></param>
	UFUNCTION(BlueprintCallable, Category = "CityBaseGen")
	void downloadconcurrently(const FString& Content);

	UFUNCTION(BlueprintCallable, Category = "CityBaseGen")
	FVector2D TestCodeToExtent(const FString& Code);

	void OnReceiveRequest(FHttpRequestPtr HttpRequest, FHttpResponsePtr HttpResponse, bool bSucceeded, FString Data);

	void InitDefaultMat();

	UPROPERTY(BlueprintAssignable)
	FOnTileDownloadFinished OnTileDownloadFinished;
	FOnGenerateFinished OnGenerateFinished;

private:
	void StartGenerate(const FString& BasePath, const FString& Code);

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Parameters")
	FCityGenParam CityGenParameters;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Parameters")
	FVector2D RealCenter;

	double Start;

};
